#!/usr/bin/env bash
curl -X POST -d '{"nickname": "noom", "email": "iam@noom.neuroops.link", "password_hash": "NxVdu9rMIdIkuViqD"}' -H 'content-type: application/json' 127.0.0.1:8081/user/signup
