#!/usr/bin/env python
# coding: utf-8

# In[379]:


import numpy as np
from sklearn.preprocessing import normalize
import mne
import statistics 
from pathlib import Path
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import pandas as pd
from mne.filter import filter_data 
from math import log
# %matplotlib inline


# In[380]:


route_baseline="/Users/polinaturiseva/Desktop/hackaton/dataset/kostya_baseline/"
# /Users/polinaturiseva/Desktop/hackaton/dataset/Kostya_zapici/music.csv
route_music="/Users/polinaturiseva/Desktop/hackaton/dataset/Kostya_zapici/"
test_base="baseline_filtered.csv"
tes_music='music.csv'
# tes_music="Muse-2F6D_2019-06-30--01-46-13_1561848571116.csv"
# /Users/polinaturiseva/Desktop/hackaton/dataset/kostya_baseline/baseline_filtered.csv


# In[381]:


def open_archive(filename, route, ch_names=None, skiprows=0, max_rows=0):
    if ch_names is None:
        ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
        
    info = mne.create_info(
        ch_names=list(ch_names.keys()),
        ch_types=['eeg' for i in range(0, len(ch_names))],
        sfreq=250,
        montage='standard_1020'
    )
    filename=route+filename
    data=np.genfromtxt(filename, delimiter=',', usecols = (0,1,2,3,4))
    data=np.delete(data, 0, 0)
    for i in range(1,5):
        data[:,i] = data[:,i]/1000
    data=data.T
    data = data[list(ch_names.values())]
    data = filter_data(data, 250, l_freq=2, h_freq=50)
    return mne.io.RawArray(data, info)
    
# sample=open_archive(filename=tes_music, route=route_music, ch_names=None, skiprows=0, max_rows=0)
# print(sample)


# In[382]:


def microvolts_to_volts(value):
    """
    Since openBCI writes data into micro volts and mne works with volts we
    will need to convert the data later.
    :param value: single micro volts value
    :return: same value in volts
    """
    return float(value) / 1000


def load_file(filename, ch_names=None, skiprows=0, max_rows=0):
    """
    Load data from file into mne RawArray for later use
    :param filename: filename for reading in form of relative path from working directory
    :param ch_names: dictionary having all or some channels like this:
            ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
            Key specifies position on head using 10-20 standard and
            Value referring to channel number on Cyton BCI board
    :return: RawArray class of mne.io library
    """
    if ch_names is None:
        ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}

    # Converter of BCI file to valuable data
    converter = {i: (microvolts_to_volts if i < 12 else lambda x: str(x).split(".")[1][:-1])
                 for i in range(0, 13)}

    info = mne.create_info(
        ch_names=list(ch_names.keys()),
        ch_types=['eeg' for i in range(0, len(ch_names))],
        sfreq=250,
        montage='standard_1020'
    )
    data = np.loadtxt(filename, comments="%", delimiter=",",
                      converters=converter, skiprows=skiprows, max_rows=max_rows).T
    data = data[list(ch_names.values())]
    data = filter_data(data, 250, l_freq=2, h_freq=50)
#     print ('data type', type(data), '  shape  ' , data.shape)
    return mne.io.RawArray(data, info)


def create_epochs(raw_data, duration=1):
    """
    Chops the RawArray onto Epochs given the time duration of every epoch
    :param raw_data: mne.io.RawArray instance
    :param duration: seconds for copping
    :return: mne Epochs class
    """
    events = mne.make_fixed_length_events(raw_data, duration=duration)
    epochs = mne.Epochs(raw_data, events, preload=True)
    return epochs


def create_epochs_with_baseline(raw_data, baseline, duration=1):
    """
    Chops the RawArray onto Epochs given the time duration of every epoch
    :param raw_data: mne.io.RawArray instance
    :param duration: seconds for copping
    :return: mne Epochs class
    """
    events = mne.make_fixed_length_events(raw_data, duration=duration)
    epochs = mne.Epochs(raw_data, events, preload=True, baseline=baseline)
    return epochs


def get_files(dir='.', pattern='*.txt'):
    """
    Loading files from given directory with specified pattern.
    :param dir: Lookup directory
    :param pattern: Pattern for files. Default *.txt for loading raw BCI files
    :return: array of file paths
    """
    # Specifying files directory, select all the files from there which is txt
    datadir = Path(dir).glob(pattern)
    # Transferring generator into array of file paths
    return [x for x in datadir]


def get_sample_data(path, regx, skiprow=100, max_row=133000):
    files = open_archive(path, regx)
    ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
    raw_data = []
    for file in files:
        raw_data.append(load_file(file, ch_names=ch_names, skiprows=skiprow, max_rows=max_row))
    real_data_series = [create_epochs(raw) for raw in raw_data]
    return real_data_series[-1]
    


# In[383]:


# TAKE ONE RANDOM RECORDING FOR PLOTTING
n_channels = 4
SAMPLE_FREQ = 250 #?


# In[384]:


def transform_ICA(sample_data):
    ica = ICA()
    ica.fit(sample_data)
    return ica.apply(sample_data) # Transform recording into ICA space


# In[385]:


def remove_epochs(data_epochs):
    dat = data_epochs.get_data()
    data = np.zeros( (dat.shape[0] * dat.shape[2], 4) )
    n_epoch = len(dat)
    n_in_epoch = dat.shape[2]
    for i in range(n_epoch):
        data[i*n_in_epoch:i*n_in_epoch + n_in_epoch] = dat[i].T
    
    return data


# In[386]:


def vectorize(sample_data, waves1, waves):
    vector = []
    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}
    
    
    ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
    
    sample_data = sample_data.get_data()
    
    analysis_data = np.zeros( (sample_data.shape[0], sample_data.shape[1]//2, sample_data.shape[2]) )
    i = 0
    
    # Calculate hemisphere difference ratio left / right
    for sample_epoch in sample_data:
        analysis_data[i][0] = (sample_epoch[1] - sample_epoch[0]) / (sample_epoch[1] + sample_epoch[0])
        analysis_data[i][1] = (sample_epoch[3] - sample_epoch[2]) / (sample_epoch[3] + sample_epoch[2])
       # analysis_data[i][2] = (sample_epoch[5] - sample_epoch[4]) / (sample_epoch[5] + sample_epoch[4])
       # analysis_data[i][3] = (sample_epoch[7] - sample_epoch[6]) / (sample_epoch[7] + sample_epoch[6])
        i+=1
    
#     vector = np.zeros((i,1))
    for epoch in analysis_data:
    
     # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(epoch.T))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(epoch.T), 1.0/SAMPLE_FREQ)
        eeg_band_fft = dict()
    
        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                               (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    
        vector.append(eeg_band_fft['Alpha'] / eeg_band_fft['Beta'])

    return np.array(vector)


# In[387]:


#  fear=0
#  mew=0
#  happy=0
#  sad=0

def vect1(sample_data):
    sample_data = sample_data.get_data()
    
    analysis_data = np.zeros( (sample_data.shape[0], sample_data.shape[1]//2, sample_data.shape[2]) )
    i = 0
    
    # Calculate hemisphere difference ratio left / right
    for sample_epoch in sample_data:
        analysis_data[i][0] = (sample_epoch[1] - sample_epoch[0]) / (sample_epoch[1] + sample_epoch[0])
        analysis_data[i][1] = (sample_epoch[3] - sample_epoch[2]) / (sample_epoch[3] + sample_epoch[2])
        #analysis_data[i][2] = (sample_epoch[5] - sample_epoch[4]) / (sample_epoch[5] + sample_epoch[4])
        #analysis_data[i][3] = (sample_epoch[7] - sample_epoch[6]) / (sample_epoch[7] + sample_epoch[6])
        i+=1
    return analysis_data
    
def vect2 (analysis_data): #alpha/beta
    vector = []
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}
    
    
    ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
    for epoch in analysis_data:
    
     # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(epoch.T))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(epoch.T), 1.0/SAMPLE_FREQ)
        eeg_band_fft = dict()
    
        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                               (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    
        vector.append(eeg_band_fft['Alpha'] / eeg_band_fft['Beta'])

    return np.array(vector)

def vect_alpha(analysis_data): # returns mean alpha for the recording??
    vector = []
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}
    
    
    ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
    for epoch in analysis_data:
    
     # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(epoch.T))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(epoch.T), 1.0/SAMPLE_FREQ)
        eeg_band_fft = dict()
    
        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                               (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    
        vector.append(eeg_band_fft['Alpha'])
    return np.array(vector)

def next_track(res):
    if log(abs(res))>3:
        return 'true'
    else:
        return 'false'

rest_threshold=1.2 

#vector is from alpha_beta
def go_to_rest(alpha_beta_start_f, alpha_betta_current_f):
    criteria=alpha_beta_start_f/alpha_betta_current_f
    if(criteria>rest_threshold):
        return 'true'
    else:
        return 'false'


def come_back_to_work(alpha_betta_current_f, alpha_beta_end_F):
    criteria= alpha_betta_current_f / alpha_beta_end_F
    if(criteria>rest_threshold):
        return 'true'
    else:
        return 'false'


# In[388]:


# from scipy import spatial 
# from sklearn.preprocessing import normalize
# meditation_vec = normalize(vectorize(sample_data, 'Alpha', 'Beta')[:,np.newaxis], axis=0).ravel()

# spatial.distance.euclidean(coding_vec, meditation_vec)


# In[389]:


#removing noize from baseline with ICA()
def baseline_prepros(route, name):
    bas=open_archive(route=route, filename=name, skiprows=10, max_rows=20000) #задержка 40 микросекунд
    bas=create_epochs(bas)
    bas=transform_ICA(bas)
    print("after ICA", type(bas))
    return bas

#returns average frequency in the baseline
def mean_baseline(route, name):
    bas=baseline_prepros(route=route, name=name)
    bas=remove_epochs(bas)
    mean = np.mean(bas, axis=0)
    return mean

#returns alpha_beta from baseline
def baseline_alpha_beta(route, name):
    bas=baseline_prepros(route=route, name=name)
    bas=normalize(vectorize(bas,'Alpha', 'Beta')[:,np.newaxis], axis=0).ravel()
    mean_alpha_beta = np.mean(bas, axis=0)
    return mean_alpha_beta


# In[390]:


# open_archive(filename, route, ch_names=None, skiprows=0, max_rows=0):

# bas=open_archive(route=route, filename=name, skiprows=10, max_rows=20000) #задержка 40 микросекунд
#     bas=create_epochs(bas)
#     bas=transform_ICA(bas)

def work_minus_bas(route,name_cod, bas_mean):
    cod=open_archive(route=route, filename=name_cod, skiprows=10, max_rows=2700)
#     print(cod)
    cod=create_epochs(cod)
#     print(cod)
    cod=transform_ICA(cod)
    cod=remove_epochs(cod)
    for i in range(4):
        cod[:,i]=cod[:, i]-bas_mean[i]
    return cod

def first_alpha_beta(work):
    ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
    info = mne.create_info(
        ch_names=list(ch_names.keys()),
        ch_types=['eeg' for i in range(0, len(ch_names))],
        sfreq=250,
        montage='standard_1020'
    )
    work=create_epochs(mne.io.RawArray(work.transpose(), info))
    work=normalize(vectorize(work,'Alpha', 'Beta')[:,np.newaxis], axis=0).ravel()
    res=np.mean(np.asarray(work))
    return res
    
def alpha_reaction(work):
    ch_names = {"af7":1, "af8":2, "tp9":3, "tp10":4}
    info = mne.create_info(
        ch_names=list(ch_names.keys()),
        ch_types=['eeg' for i in range(0, len(ch_names))],
        sfreq=250,
        montage='standard_1020'
    )
    work=create_epochs(mne.io.RawArray(work.transpose(), info))
    work=normalize((vect_alpha(work))[:,np.newaxis], axis=0).ravel()
    res=np.mean(np.asarray(work))
    return res


# In[391]:


def hemispheres(data):
    dif=np.zeros((2,1))
    #fp_dif
    dif[0,0]= np.mean(np.asarray(data[0]))-np.mean(np.asarray(data[3]))
    #o_dif
    dif[1,0]= np.mean(np.asarray(data[2]))-np.mean(np.asarray(data[1]))
    #c_dif
    return dif

def stress(data):
    data=np.mean(data, axis=0)
#     print ("mew", data)
#     print("shape", data.shape)
    a=(max(data[0],data[3]))/(min(data[0],data[3]))
    b=(max(data[1],data[2]))/(min(data[1],data[2]))
    res=(a+b)/2
    if res>0.2:
        return 'true'
    else:
        return 'false'


# In[392]:


#код обработки
#0-3 correlate to TP9, Fp1, Fp2 and TP10 respectively.
#replay=3
# work_minus_bas(route,name_cod, bas_mean):
# TODO: do not forget
# base_mean=mean_baseline(route=route_baseline, name=test_base)
# bas_alpha=baseline_alpha(route=route_music, name=tes_music)


# TODO: Move into function

# tes_music - имя файла 10с сэмпла
# route_music - путь до сэмплов
def process_sample(base_mean, sample_filename, path_to_samples, is_first=True):
    start='true' # сделать так, чтобы мы получали это извне как сигнал на старт записи
    start_alpha_beta = None

    working = work_minus_bas(route=path_to_samples,
                             name_cod=sample_filename,
                             bas_mean=base_mean)
    if start=='true':
        start_alpha_beta=work_minus_bas(route=path_to_samples,
                                        name_cod=sample_filename,
                                        bas_mean=base_mean)
        rest='false'
    else:
        alpha_betta_cur = first_alpha_beta(working)
        rest=go_to_rest(start_alpha_beta_f=start_alpha_beta,
                        alpha_betta_current_f=alpha_betta_cur)  # boolean который надо

    # alpha_react = alpha_reaction(working)

    # print(alpha_react)
    further = next_track(alpha_reaction(working)) # boolean который надо вернуть
    difference = hemispheres(working)
    strs = stress(working)#boolean который надо вернуть
    return { 'rest': rest,
             'further': further,
             'stress': strs,
             'difference': difference,
             'start_alpha_beta': start_alpha_beta
             }

# if rest == true then go to relax
# TODO: add call to comeback function

# if further == true then call next_track
# difference - float[] - сохранить в базу
# stress - boolean - сохранить в базу
