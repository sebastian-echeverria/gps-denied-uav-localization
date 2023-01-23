#!/bin/bash
docker run -it -v "$(pwd)/data":/app/data gpsd $@
