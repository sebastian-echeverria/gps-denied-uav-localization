from __future__ import annotations

from math import ceil

import torch
from torch import Tensor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def open_image_as_tensor(img_path: str, target_height: int=0) -> Tensor:
	# Opens an image as a tensor.
	# img_path: full path to image file.
	# target_size: potential target size to resize the image to. Value of 0 means to not resize.
	#
	# Returns an image as a Tensor (with 3-Ds, as a pixel matrix inside an array).

	# Convert to RGB to ensure the image will have 3 channels.
	img = Image.open(img_path).convert('RGB')

	# If the parameter indicates it, resize image.
	if target_height != 0:
		img_w, img_h = img.size
		aspect = img_w / img_h
		img_h_sm = target_height
		img_w_sm = ceil(aspect * img_h_sm)
		img = img.resize((img_w_sm, img_h_sm))

	# Convert image to tensor.
	img_tens = convert_image_to_tensor(img)
	print(img_tens.size())
	img_tens = torch.unsqueeze(img_tens, 0)
	print(img_tens.size())	
	return img_tens


def save_tensor_image_to_file(image_tensor: Tensor, file_path: str):
	# Saves the given tensor image into a file.
	image = convert_tensor_to_image(image_tensor)
	plt.imsave(file_path, image)


def convert_image_to_tensor(img: Image) -> Tensor:
	# Converts a PIL image into a Tensor.
	# img: the PIL image.
	#
	# Returns a Tensor with the PIL image data.
	convert_to_tensor = transforms.ToTensor()
	return convert_to_tensor(img)


def convert_tensor_to_image(img_tensor: Tensor) -> Image:
	# Converts a Tensor to a PIL image.
	# img_tensor: the image as a tensor.
	#
	# Returns an PIL Image with the same info.
	convert_to_PIL_image = transforms.ToPILImage()
	return convert_to_PIL_image(img_tensor)
