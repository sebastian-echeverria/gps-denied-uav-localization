#!/bin/bash
docker run -it --mount type=bind,source="$(pwd)/data"/target,target=/app/data \
gpsd $@
