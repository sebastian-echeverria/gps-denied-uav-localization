from __future__ import annotations

import argparse
from typing import Tuple

import torch
from torch import Tensor
import numpy.typing as npt

import image_io
import DeepLKBatch as dlk
import pix2coords
from pix2coords import ImageProjector
import sift

# suppress endless SourceChangeWarning messages from pytorch
import warnings
warnings.filterwarnings("ignore")

USE_CUDA = torch.cuda.is_available()


def load_dlk_net(model_path: str) -> dlk.DeepLK:
    # Loads the DLK network we will use.
    return dlk.DeepLK(dlk.custom_net(model_path))


def calculate_homography_from_model(sat_image: Tensor, uav_image: Tensor, model_path: str) -> Tuple[Tensor, Tensor]:
    # Calculates the P array for an homography matrix given 2 images and a model to be used by the DeepLK algorithm.
    # Returns a Tensor with the P array.

    print("Loading DLK net and model...")
    dlk_net = load_dlk_net(model_path)
    print("Done loading net.")

    print("Executing DLK net on both images to get motion parameters.")
    # NOTE: we should be passing the sat image first, and the uav/template image second, according to the DLK params. But in their main test, they switch them...
    p_lk, homography = dlk_net(uav_image, sat_image, tol=1e-2, max_itr=200, conv_flag=1)
    inv_p = dlk_net.get_inverse_p(p_lk)
    print("DLK execution ended.")

    return p_lk, homography, inv_p


def warp_image(input_tensor: Tensor, template_tensor: Tensor, homography_tensor: Tensor) -> Tensor:
    """Gets a input and template image as tensors, along with a homography, and returns a warped image tensor."""
    # Drop the batch and channel dimensions from the tensors, since they are not needed for the warping.
    input_image = input_tensor[0, 0, :, :].numpy()
    template_image = template_tensor[0, 0, :, :].numpy()

    homography_numpy = homography_tensor.squeeze(0).detach().numpy()
    warped_image = pix2coords.warp_image(input_image, template_image, homography_numpy) 
    return image_io.convert_image_to_tensor(warped_image)   


def calculate_coordinates(input_path: str, template_path: str, homography_tensor: Tensor) -> npt.NDArray:
    """Calculates coordinates given the homography."""
    homography_numpy = homography_tensor.squeeze(0).detach().numpy()
    projector = ImageProjector()
    projector.load_template_image(template_path)
    projector.load_input_image(input_path)
    projector.homography = homography_numpy
    gps_coords, _, _ = projector.infer_coordinates()
    return gps_coords


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

    # 1. Load images properly (load them, normalize them, etc).
    print("Loading zone image...")
    sat_image = image_io.open_image_as_tensor(args.SAT_PATH)
    print("Loading UAV image...")
    uav_image = image_io.open_image_as_tensor(args.PIC_PATH)
    print("Images loaded")

    # 2. Run the dlk_trained on these two images (it receives two batches, but in this case each batch will be of 1)
    # to get the params and homography matrix from dlk.
    print("Calculating homography using Goforth algorithm...")
    p, homography, inv_p = calculate_homography_from_model(sat_image, uav_image, args.MODEL_PATH)
    print("Finished calculating homography from algorithm.")

    # 3. Use matrix to apply it to one image, and store results for visual inspection.
    print(f"UAV Image size: {uav_image.shape}")
    print(f"SAT Image size: {sat_image.shape}")
    print("Extract projection from SAT image that should match UAV image...")
    projected_image_2,_ = dlk.get_input_projection_for_template(uav_image, sat_image, p)
    projected_image_3,_ = dlk.get_input_projection_for_template(sat_image, uav_image, inv_p)
    print(f"Finished projecting; projected Image size: {projected_image_2.shape}")
    image_io.save_tensor_image_to_file(projected_image_2, "./data/p1-uav-warped-to-map-angles.png")
    image_io.save_tensor_image_to_file(projected_image_3, "./data/p2-map-extraction-projected-uav.png")
    print("Projected image saved to disk.")

    # 4. Convert to GPS coordinates.
    gps_coords = calculate_coordinates(args.SAT_PATH, args.PIC_PATH, homography)
    print(f"Coordinates from Goforth homography: {gps_coords}")

    # Extra: Baseline with SIFT for comparison.
    print("As baseline: use SIFT to extract projection from SAT image that should match UAV image...")
    projector, gps_coords = sift.align_and_show(args.PIC_PATH, args.SAT_PATH)
    projected_image = image_io.convert_image_to_tensor(projector.get_input_projection_for_template())
    print(f"Projected Image 2 size: {projected_image.shape}")
    image_io.save_tensor_image_to_file(projected_image, "./data/projected2.png")
    print("Baseline projected image saved to disk.")
    print(f"SIFT coords: {gps_coords}")


# Entry hook.
if __name__ == "__main__":
    main()
