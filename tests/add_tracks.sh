#!/usr/bin/env bash

curl -X PUT -d @track01.json -H 'content-type: application/json' 127.0.0.1:8081/tracks/add