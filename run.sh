#!/bin/bash
echo $@
docker run -it -v "$(pwd)/data":/app/data gpsd run_local.sh $@
