#!/usr/bin/env python3
import datetime
import logging
import os
import hackatonchik as eeg_processing
import tempfile

import connexion
from connexion import NoContent

import orm

db_session = None
user_id = 1

# def get_pets(limit, animal_type=None):
#     q = db_session.query(orm.Pet)
#     if animal_type:
#         q = q.filter(orm.Pet.animal_type == animal_type)
#     return [p.dump() for p in q][:limit]


def post_signup(body):
    user = orm.User(nickname=body.get('nickname', None),
             email=body.get('email', None),
             password_hash=body.get('password_hash', None))

    db_session.add(user)
    db_session.commit()

    return user.dump()


# TODO: Do auth
def post_signin(**kwargs):
    user = db_session.query(orm.User).filter(orm.User.id == user_id).one_or_none()
    return


# TODO: Do auth and get user_id from params
def get_userdata(**kwargs):
    user = db_session.query(orm.User).filter(orm.User.id == user_id).one_or_none()
    return user.dump() if user is not None else ('Not found', 404)


# TODO: filter by current user_id over wrapper
def put_track_add(body):
    # TODO: Implement it
    # look good
    new_track = orm.Track(**body, user_id=user_id)
    db_session.add(new_track)
    print(new_track)
    return new_track.dump()


# TODO: filter by current user_id over wrapper
def get_next_track():
    next_track = db_session.query(orm.Track).first()
    return next_track.dump() if next_track is not None else ('Not found', 404)


# TODO: filter by current user_id over wrapper
def put_track_update(body):
    # TODO: Implement it
    print(body)
    return


# TODO: filter by current user_id over wrapper
def baseline_put(baseline):
    # TODO: put file to tmp dir
    # TODO:
    path_to_tmp_dir = tempfile.gettempdir()
    tmp_file_name = tempfile.TemporaryFile(mode="w", dir=path_to_tmp_dir)
    mean = eeg_processing.mean_baseline(path_to_tmp_dir, tmp_file_name)
    baseline_obj = orm.UserBaseline
    # TODO: Save baseline data to db
    return 201


# TODO: filter by current user_id over wrapper
def put_track_history(track_id, body):

    track = db_session.query(orm.Track).filter(orm.Track.id == track_id).one_or_none()
    if track:
        track_history = orm.TrackHistory(track_id=track.id, **body)
        db_session.add(track_history)
        return track_history.dump()
    else:
        return 'Not found', 404
    # return track.dump() if track is not None else ('Not found', 404)


# TODO: filter by current user_id over wrapper
def get_track(track_id):
    track = db_session.query(orm.Track).filter(orm.Track.id == track_id).one_or_none()
    return track.dump() if track is not None else ('Not found', 404)


db_connstring = os.environ.get('NEURO_CONNSTRING', 'sqlite:///:memory:')
logging.basicConfig(level=logging.INFO)
db_session = orm.init_db(db_connstring)
app = connexion.FlaskApp(__name__)
app.add_api('openapi.yaml')

application = app.app


@application.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()


if __name__ == '__main__':
    app.run(port=8081, use_reloader=True, threaded=False)
