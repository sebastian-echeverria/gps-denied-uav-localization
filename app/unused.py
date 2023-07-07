

from torch import Tensor

import image_io
import pix2coords

def warp_image(input_tensor: Tensor, template_tensor: Tensor, homography_tensor: Tensor) -> Tensor:
    """Gets a input and template image as tensors, along with a homography, and returns a warped image tensor."""
    # Drop the batch and channel dimensions from the tensors, since they are not needed for the warping.
    input_image = input_tensor[0, 0, :, :].numpy()
    template_image = template_tensor[0, 0, :, :].numpy()

    homography_numpy = homography_tensor.squeeze(0).detach().numpy()
    warped_image = pix2coords.warp_image(input_image, template_image, homography_numpy) 
    return image_io.convert_image_to_tensor(warped_image)
