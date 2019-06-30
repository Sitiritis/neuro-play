from sqlalchemy.dialects import postgresql
from sqlalchemy import Column, DateTime, String, Integer, Text, ForeignKey, ForeignKeyConstraint, \
    Sequence, Float, dialects, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.orm import scoped_session, sessionmaker


Base = declarative_base()


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    nickname = Column(String(32), index=True, nullable=False)
    _password_hash = Column('password_hash', String(128), nullable=False)
    email = Column(Text, nullable=False)
    type = Column(String(20))
    created = Column(DateTime(), default=datetime.utcnow)
    tracks = relationship("Track")

    def dump(self):
        return dict([(k, v) for k, v in vars(self).items() if not k.startswith('_')])


class Track(Base):
    __tablename__ = 'user_track'
    id = Column(Integer, Sequence('user_track_id_seq'), primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    track_name = Column(Text, nullable=False)
    android_meta = Column(postgresql.JSONB(), nullable=True)
    duration = Column(Integer, nullable=False)
    add_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # track_history = Column(Integer, ForeignKey('user.id'), nullable=False)
    track_history = relationship("TrackHistory")

    def dump(self):
        return dict([(k, v) for k, v in vars(self).items() if not k.startswith('_')])


class TrackHistory(Base):
    __tablename__ = 'user_track_history'
    id = Column(Integer, Sequence('user_track_history_id_seq'), primary_key=True)
    add_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_track_id = Column(Integer, ForeignKey('user_track.id'), nullable=False)
    # listen_duration = Column(Float, nullable=False) # TODO: Get it from samples
    # productivity_parameter = Column(Float, nullable=False) # TODO: Compute it from samples


    def dump(self):
        return dict([(k, v) for k, v in vars(self).items() if not k.startswith('_')])


class TrackSessionSample(Base):
    __tablename__ = 'track_samples'
    id = Column(Integer, Sequence('user_track_history_id_seq'), primary_key=True)
    duration = Column(Float, nullable=False)
    timestamp = Column(dialects.postgresql.TIMESTAMP(precision=3), nullable=False)
    eegdata = Column(postgresql.ARRAY(Float), nullable=False)
    stress = Column(postgresql.BOOLEAN, default=False)


class UserBaseline(Base):
    __tablename__ = 'user_baseline'
    id = Column(Integer, Sequence('user_baseline_id_seq'), primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    # duration = Column(Float, nullable=False) # Get it from EEG data
    eegdata = Column(postgresql.ARRAY(Float), nullable=False)
    mean = Column(postgresql.ARRAY(Float), nullable=False)


def init_db(uri):
    engine = create_engine(uri, convert_unicode=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    Base.query = db_session.query_property()
    # Base.metadata.drop_all(bind=engine) # FIXME: Terrible solution!
    Base.metadata.create_all(bind=engine)

    return db_session
