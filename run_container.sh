#!/bin/bash
echo $@
docker run --rm -it -v "$(pwd)/data":/app/data gpsd $@
