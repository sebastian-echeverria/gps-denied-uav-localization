import argparse

import torch
from torch.autograd import Variable

from image_handler import open_image_as_tensor

USE_CUDA = torch.cuda.is_available()

def load_image(image_path: str):
    image = Variable(torch.zeros(num_minibatch, 3, training_sz, training_sz))
    if USE_CUDA:
        image = image.cuda()


def main():
    # STEPS
    # 0. LATER: Add additional steps to get an image close to the picture from the map, given assumptions on where we are.
    # 1. Obtain 2 similar images (created manually, externally from this app), as inputs
    # 2. Load images properly (load them, normalize them, etc).
    # 3. Run the dlk_trained on these two images (it receives two batches, but in this case each batch will be of 1)
    # 4. Check out the params and homography matrix from dlk
    # 5. Use matrix to apply it to one image, and somehow store the modified image to see resutlts?
    # 6. LATER: convert to GPS coordinates.

    parser = argparse.ArgumentParser()
    parser.add_argument("SAT_PATH")
    parser.add_argument("PIC_PATH")
    parser.add_argument("MODEL_PATH")
    args = parser.parse_args()

    print(f"Satellite image path: {args.SAT_PATH}")
    print(f"UAV photo image path: {args.PIC_PATH}")
    print(f"CNN model path: {args.MODEL_PATH}")

    sat_image = open_image_as_tensor(args.SAT_PATH)
    uav_image = open_image_as_tensor(args.PIC_PATH)
    


# Entry hook.
if __name__ == "__main__":
    main()
