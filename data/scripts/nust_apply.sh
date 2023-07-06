#!/bin/bash
cd ../..
#bash run_container.sh run_apply.sh ./data/sat_data/nust/nust_zoom1.tif ./data/sat_data/nust/Image844.jpg ./data/models/conv_03_13_18_1850.pth


bash run_container.sh run_apply.sh ./data/sat_data/nust/nust_zoom3.tif ./data/sat_data/nust/Image844-rotated2.png ./data/models/conv_03_13_18_1850.pth
#bash run_container.sh run_apply.sh ./data/sat_data/nust/Image844-rotated.jpg ./data/sat_data/nust/Image844.jpg ./data/models/conv_03_13_18_1850.pth

# Rotated, full same image
#bash run_container.sh run_apply.sh ./data/input.jpg ./data/input-rotated.jpg ./data/models/conv_03_13_18_1850.pth

# Working, slightly rotated crop of same image.
#bash run_container.sh run_apply.sh ./data/input.jpg ./data/test3-input2-warped-initial-p.jpg ./data/models/conv_03_13_18_1850.pth
cd data/scripts
