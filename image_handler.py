import os
import random
import glob
from math import cos, sin, pi, ceil

import torch
from torch import Tensor
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

import DeepLKBatch as dlk


# size scale range
min_scale = 0.75
max_scale = 1.25

# rotation range (-angle_range, angle_range)
angle_range = 15 # degrees

# projective variables (p7, p8)
projective_range = 0

# translation (p3, p6)
translation_range = 10 # pixels

# possible segment sizes
lower_sz = 200 # pixels, square
upper_sz = 220

USE_CUDA = torch.cuda.is_available()


def generate_image_pairs(sat_path, batch_size, training_sz, training_sz_pad, warp_pad):
	# create batch of normalized training pairs

    # sat_path: path to folder with images/ subfolder with multiple versions of a satellite map image.
	# batch_size [in, int] : number of pairs
    #
	# img_batch [out, Tensor N x 3 x training_sz x training_sz] : batch of images
	# template_batch [out, Tensor N x 3 x training_sz x training_sz] : batch of templates
	# param_batch [out, Tensor N x 8 x 1] : batch of ground truth warp parameters

	print('min_scale: ',  min_scale)
	print('max_scale: ', max_scale)
	print('angle_range: ', angle_range)
	print('projective_range: ', projective_range)
	print('translation_range: ', translation_range)
	print('lower_sz: ', lower_sz)
	print('upper_sz: ', upper_sz)

	# randomly choose 2 aligned images
	FOLDERPATH = os.path.join(sat_path, 'images/')
	images_dir = glob.glob(FOLDERPATH + '*.png')
	random.shuffle(images_dir)

	img = Image.open(images_dir[0])
	template = Image.open(images_dir[1])

	in_W, in_H = img.size

	# pdb.set_trace()

	# initialize output tensors

	if USE_CUDA:
		img_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz)).cuda()
		template_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz)).cuda()
		param_batch = Variable(torch.zeros(batch_size, 8, 1)).cuda()
	else:
		img_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz))
		template_batch = Variable(torch.zeros(batch_size, 3, training_sz, training_sz))
		param_batch = Variable(torch.zeros(batch_size, 8, 1))


	for i in range(batch_size):

		# randomly choose size and top left corner of image for sampling
		seg_sz = random.randint(lower_sz, upper_sz)
		seg_sz_pad = round(seg_sz + seg_sz * 2 * warp_pad)

		loc_x = random.randint(0, (in_W - seg_sz_pad) - 1)
		loc_y = random.randint(0, (in_H - seg_sz_pad) - 1)

		img_seg_pad = img.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
		img_seg_pad = img_seg_pad.resize((training_sz_pad, training_sz_pad))

		template_seg_pad = template.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
		template_seg_pad = template_seg_pad.resize((training_sz_pad, training_sz_pad))

		if USE_CUDA:
			img_seg_pad = Variable(transforms.ToTensor()(img_seg_pad).cuda())
			template_seg_pad = Variable(transforms.ToTensor()(template_seg_pad).cuda())
		else:
			img_seg_pad = Variable(transforms.ToTensor()(img_seg_pad))
			template_seg_pad = Variable(transforms.ToTensor()(template_seg_pad))

		# create random ground truth
		scale = random.uniform(min_scale, max_scale)
		angle = random.uniform(-angle_range, angle_range)
		projective_x = random.uniform(-projective_range, projective_range)
		projective_y = random.uniform(-projective_range, projective_range)
		translation_x = random.uniform(-translation_range, translation_range)
		translation_y = random.uniform(-translation_range, translation_range)

		rad_ang = angle / 180 * pi

		if USE_CUDA:
			p_gt = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
									   -sin(rad_ang),
									   translation_x,
									   sin(rad_ang),
									   scale + cos(rad_ang) - 2,
									   translation_y,
									   projective_x, 
									   projective_y]).cuda())
		else:
			p_gt = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
									   -sin(rad_ang),
									   translation_x,
									   sin(rad_ang),
									   scale + cos(rad_ang) - 2,
									   translation_y,
									   projective_x, 
									   projective_y]))

		p_gt = p_gt.view(8,1)
		p_gt = p_gt.repeat(1,1,1)

		p_gt_H = dlk.param_to_H(p_gt)
		inv_result = dlk.InverseBatch.apply(p_gt_H)
		img_seg_pad_w, _ = dlk.warp_hmg(img_seg_pad.unsqueeze(0), dlk.H_to_param(inv_result))

		img_seg_pad_w.squeeze_(0)

		pad_side = round(training_sz * warp_pad)

		img_seg_w = img_seg_pad_w[:,
							pad_side : pad_side + training_sz,
							pad_side : pad_side + training_sz]



		template_seg = template_seg_pad[:,
							pad_side : pad_side + training_sz,
							pad_side : pad_side + training_sz]

		img_batch[i, :, :, :] = img_seg_w
		template_batch[i, :, :, :] = template_seg

		param_batch[i, :, :] = p_gt[0, :, :].data

		# transforms.ToPILImage()(img_seg_w.data[:, :, :]).show()
		# time.sleep(2)
		# transforms.ToPILImage()(template_seg.data[:, :, :]).show()

		# print('angle: ', angle)
		# print('scale: ', scale)
		# print('proj_x: ', projective_x)
		# print('proj_y: ', projective_y)
		# print('trans_x: ', translation_x)
		# print('trans_y: ', translation_y)

		# pdb.set_trace()

	return img_batch, template_batch, param_batch


def open_image_as_tensor(img_path: str, target_height: int=0) -> Tensor:
	# Opens an image as a tensor.
	# img_path: full path to image file.
	# target_size: potential target size to resize the image to. Value of 0 means to not resize.
	#
	# Returns an image as a Tensor (with 3-Ds, as a pixel matrix inside an array).

	preprocess = transforms.Compose([
		transforms.ToTensor(),
	])

	img = Image.open(img_path)

	# If the parameter indicates it, resize image.
	if target_height != 0:
		img_w, img_h = img.size
		aspect = img_w / img_h
		img_h_sm = target_height
		img_w_sm = ceil(aspect * img_h_sm)
		img = img.resize((img_w_sm, img_h_sm))

	# Convert image to tensor.
	img_tens = preprocess(img)
	img_tens = torch.unsqueeze(img_tens, 0)
	return img_tens


def convert_tensor_to_image(img_tensor: Tensor) -> Image:
	# Converts a Tensor to a PIL image.
	# img_tensor: the image as a tensor.
	#
	# Returns an PIL Image with the same info.
	transform = transforms.ToPILImage()
	img = transform(img_tensor)
	return img