#!/bin/bash
cd ../..
bash run_container.sh run_test.sh ./data/sat_data/woodbridge/ ./data/models/conv_03_13_18_1850.pth ./data/models/vgg16_model.pth -t ./data/params.txt
cd data/scripts
