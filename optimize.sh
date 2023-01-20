#!/bin/bash
pipenv run python3 pose_opt.py sliding_window \
-image_dir ./data/village/frames/ \
-image_dir_ext *.JPG \
-motion_param_loc ./data/village/P_village.csv \
-map_loc ./data/village/map_village.jpg \
-model_path ./data/models/conv_02_17_18_1833.pth \
-opt_img_height 100 \
-img_h_rel_pose 1036.8 \
-opt_param_save_loc ./data/village/test_out.mat
