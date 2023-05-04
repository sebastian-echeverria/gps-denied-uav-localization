from __future__ import annotations

import argparse

import torch
from torch import Tensor

import image_io
import image_processor
import DeepLKBatch as dlk
import pix2coords

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
    p_lk, _ = dlk_net(sat_image, uav_image, tol=1e-4, max_itr=50, conv_flag=1)    
    print("DLK execution ended.")

    return p_lk


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
    p = calculate_homography_from_model(sat_image, uav_image, args.MODEL_PATH)

    # 5. Use matrix to apply it to one image, and somehow store the modified image to see results?
    # Project image.
    projected_image, _ = image_processor.project_images(uav_image, p)

    # Save to file.
    image_io.save_tensor_image_to_file(projected_image, "./data/projected.png")

    # 6. LATER: convert to GPS coordinates.
    # TODO: Need to have cropped GeoTIFF image with coordinates for this to work.
    homography = image_processor.param_to_H(p)
    #pix2coords.infer_coordinates_from_paths(args.PIC_PATH, args.SAT_PATH, homography)


# Entry hook.
if __name__ == "__main__":
    main()
