from __future__ import annotations

import argparse

import torch
from torch import Tensor

import image_io
import image_processor
import DeepLKBatch as dlk

USE_CUDA = torch.cuda.is_available()


def calculate_homography_from_model(tensor_image_1: Tensor, tensor_image_2: Tensor, model_path: str) -> Tensor:
    # Calculates the P array for an homography matrix given 2 images and a model to be used by the DeepLK algorithm.
    # Returns a Tensor with the P array.

    #M_tmpl_tens = Variable(torch.from_numpy(tensor_image_1).float())
    M_tmpl_tens_nmlz = dlk.normalize_img_batch(tensor_image_1)
        
    #T_tmpl_tens = Variable(torch.from_numpy(tensor_image_2).float())
    T_tmpl_tens_nmlz = dlk.normalize_img_batch(tensor_image_2)        

    dlk_net = dlk.DeepLK(dlk.custom_net(model_path))
    p_lk, _ = dlk_net(M_tmpl_tens_nmlz, T_tmpl_tens_nmlz, tol=1e-4, max_itr=50, conv_flag=1)    
    
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
    sat_image = image_io.open_image_as_tensor(args.SAT_PATH)
    uav_image = image_io.open_image_as_tensor(args.PIC_PATH)

    # 3. Run the dlk_trained on these two images (it receives two batches, but in this case each batch will be of 1)
    # 4. Check out the params and homography matrix from dlk
    p = calculate_homography_from_model(sat_image, uav_image, args.MODEL_PATH)

    # 5. Use matrix to apply it to one image, and somehow store the modified image to see resutlts?
    # Project image.
    projected_images = image_processor.project_images(uav_image, p)
    projected_image = projected_images[0,:,:,:]

    # Save to file.
    image_io.save_tensor_image_to_file(projected_image, "./data/projected.png")

    # 6. LATER: convert to GPS coordinates.
    # TODO: Convert to coordinates.


# Entry hook.
if __name__ == "__main__":
    main()
