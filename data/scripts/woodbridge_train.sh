#!/bin/bash
cd ../..
bash run_container.sh gdul_train.sh ./data/sat_data/woodbridge/ ./data/models/trained_model_output.pth ./data/models/vgg16_model.pth
cd data/scripts
