from __future__ import annotations

import argparse

import torch
from torch import Tensor

import image_io
import image_processor
import DeepLKBatch as dlk
import pix2coords
import sift

# suppress endless SourceChangeWarning messages from pytorch
import warnings
warnings.filterwarnings("ignore")

USE_CUDA = torch.cuda.is_available()


def load_dlk_net(model_path: str) -> dlk.DeepLK:
    # Loads the DLK network we will use.
    return dlk.DeepLK(dlk.custom_net(model_path))


def calculate_homography_from_model(sat_image: Tensor, uav_image: Tensor, model_path: str) -> Tensor:
    # Calculates the P array for an homography matrix given 2 images and a model to be used by the DeepLK algorithm.
    # Returns a Tensor with the P array.

    print("Loading DLK net and model...")
    dlk_net = load_dlk_net(model_path)
    print("Done loading net.")

    print("Executing DLK net on both images to get motion parameters.")
    p_lk, homography = dlk_net(sat_image, uav_image, tol=1e-4, max_itr=50, conv_flag=1)    
    print("DLK execution ended.")

    return p_lk, homography


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("SAT_PATH")
    parser.add_argument("PIC_PATH")
    parser.add_argument("MODEL_PATH")
    args = parser.parse_args()

    print(f"Satellite image path: {args.SAT_PATH}")
    print(f"UAV photo image path: {args.PIC_PATH}")
    print(f"CNN model path: {args.MODEL_PATH}")

    # 0. LATER: Add additional steps to get an image close to the picture from the map, given assumptions on where we are.
    # TODO: Steps to zoom into image from map.

    # 1. Obtain 2 similar images (created manually, externally from this app), as inputs
    # 2. Load images properly (load them, normalize them, etc).
    print("Loading zone image...")
    sat_image = image_io.open_image_as_tensor(args.SAT_PATH)
    print("Loading UAV image...")
    uav_image = image_io.open_image_as_tensor(args.PIC_PATH)
    print("Images loaded")

    # 3. Run the dlk_trained on these two images (it receives two batches, but in this case each batch will be of 1)
    # 4. Check out the params and homography matrix from dlk
    p, homography = calculate_homography_from_model(sat_image, uav_image, args.MODEL_PATH)
    print(p)

    # 5. Use matrix to apply it to one image, and somehow store the modified image to see results?
    # Project image and save to file.
    # TODO: check how to really test p
    print(f"UAV Image size: {uav_image.shape}")
    print(f"SAT Image size: {sat_image.shape}")
    projected_image, _ = dlk.get_input_projection_for_template(sat_image, uav_image, p)
    print(f"Projected Image size: {projected_image.shape}")
    image_io.save_tensor_image_to_file(projected_image, "./data/projected.png")

    projector, gps_coords = sift.align_and_show(args.PIC_PATH, args.SAT_PATH)

    # 6. LATER: convert to GPS coordinates.
    #gps_coords, _, _ = pix2coords.infer_coordinates_from_paths(args.PIC_PATH, args.SAT_PATH, homography)
    #print(gps_coords)
    h_tensor = torch.from_numpy(projector.homography)
    h_tensor.unsqueeze_(0)
    p = dlk.H_to_param(h_tensor).float()
    print(p)
    projected_image, _ = dlk.get_input_projection_for_template(sat_image, uav_image, p)
    print(f"Projected Image 2 size: {projected_image.shape}")
    image_io.save_tensor_image_to_file(projected_image, "./data/projected2.png")


# Entry hook.
if __name__ == "__main__":
    main()
