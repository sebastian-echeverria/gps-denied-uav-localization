from __future__ import annotations

import argparse

from torch import Tensor
import numpy.typing as npt

import image_io
import DeepLKBatch as dlk
import pix2coords
from pix2coords import ImageProjector
import sift
import mask_corners


def calculate_projected_corners(uav_image: Tensor, sat_image: Tensor, p: Tensor, inv_p: Tensor) -> npt.NDArray:
    """Projects the images in both ways, and returns the corners of the first projection."""
    print(f"UAV Image size: {uav_image.shape}")
    print(f"SAT Image size: {sat_image.shape}")
    print("Extract projection from SAT image that should match UAV image...")

    projected_image, mask = dlk.calculate_projection(uav_image, sat_image, p)
    print(f"Finished projecting; projected Image size: {projected_image.shape}")

    projected_image_inv, _ = dlk.calculate_projection(sat_image, uav_image, inv_p)
    print(f"Finished projecting; inverse projected Image size: {projected_image_inv.shape}")

    # Save projections to disk for visual inspection.
    image_io.save_tensor_image_to_file(projected_image, "./data/p1-uav-warped-to-map-angles.png")
    image_io.save_tensor_image_to_file(projected_image_inv, "./data/p2-map-extraction-projected-uav.png")
    print("Projected images saved to disk.")
    
    # Calculate the projected corners from the mask.
    projected_corners = mask_corners.find_corners(mask.squeeze(0).numpy())
    return projected_corners


def calculate_coordinates_from_projection(input_path: str, template_path: str, projected_corners: npt.NDArray) -> npt.NDArray:
    """Calculates coordinates given two images and a mask."""
    #homography_numpy = homography_tensor.squeeze(0).detach().numpy()
    projector = ImageProjector()
    projector.load_template_image(template_path)
    projector.load_input_image(input_path)
    gps_coords, _ = projector.infer_coordinates(projected_corners)
    pix2coords.show_corners(projector.input_image, projected_corners, "data/goforthcorners.png")
    return gps_coords


def calculate_sift_baseline(uav_pic_path, sat_map_path):
    """Calculates and stores the SIFT alignment image, to be used as comparison."""
    print("As baseline: use SIFT to extract projection from SAT image that should match UAV image...")
    projector, gps_coords = sift.align_and_show(uav_pic_path, sat_map_path)
    projected_image = image_io.convert_image_to_tensor(projector.get_input_projection_for_template())
    print(f"Projected Image 2 size: {projected_image.shape}")
    image_io.save_tensor_image_to_file(projected_image, "./data/projected2.png")
    print("Baseline projected image saved to disk.")
    print(f"SIFT coords: {gps_coords}")


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

    # 1. Load images as tensors.
    print("Loading zone image...")
    sat_image = image_io.open_image_as_tensor(args.SAT_PATH)
    print("Loading UAV image...")
    uav_image = image_io.open_image_as_tensor(args.PIC_PATH)
    print("Images loaded")

    # 2. Run the CNNs on the images, and use the DLK algorithm to get the homography params.
    print("Calculating homography using Goforth algorithm...")
    p, homography, inv_p, inv_homography = dlk.calculate_homography_from_model(sat_image, uav_image, args.MODEL_PATH)
    print("Finished calculating homography from algorithm.")

    # 3. Using the homography params, project image and get mask with projection corners.
    projected_corners = calculate_projected_corners(uav_image, sat_image, p, inv_p)

    # 4. Convert to GPS coordinates from mask, assuming mask represents the corners of the projected image.
    gps_coords = calculate_coordinates_from_projection(args.SAT_PATH, args.PIC_PATH, projected_corners)
    print(f"Coordinates from Goforth homography: {gps_coords}")

    # Extra: Baseline with SIFT for comparison.
    calculate_sift_baseline(args.PIC_PATH, args.SAT_PATH)


# Entry hook.
if __name__ == "__main__":
    main()
