#!/bin/bash
cd ../..
#bash run_container.sh gdul_apply.sh ./data/sat_data/nust/nust_zoom2.tif ./data/sat_data/nust/Image844.jpg ./data/models/conv_03_13_18_1850.pth
#bash run_container.sh gdul_apply.sh ./data/sat_data/nust/Image844-rotated.jpg ./data/sat_data/nust/Image844.jpg ./data/models/conv_03_13_18_1850.pth
bash run_container.sh gdul_apply.sh ./data/input.jpg ./data/input-rotated.jpg ./data/models/conv_03_13_18_1850.pth
#bash run_container.sh gdul_apply.sh ./data/test1-input1.jpg ./data/test3-input2-warped-initial-p.jpg ./data/models/conv_03_13_18_1850.pth
cd data/scripts
