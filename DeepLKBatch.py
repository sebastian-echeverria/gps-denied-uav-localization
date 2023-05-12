import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import grid_sample

from utils import printd

USE_CUDA = torch.cuda.is_available()

class InverseBatch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        batch_size, h, w = input.size()
        assert(h == w)
        H = torch.Tensor(batch_size, h, h).type_as(input)
        for i in range(0, batch_size):
            H[i, :, :] = input[i, :, :].inverse()
        ctx.save_for_backward(H) 
        #self.H = H

        return H

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output.is_contiguous())
        H, = ctx.saved_tensors
        #H = self.H

        [batch_size, h, w] = H.size()
        assert(h == w)
        Hl = H.transpose(1,2).repeat(1, 1, h).view(batch_size*h*h, h, 1)
        # print(Hl.view(batch_size, h, h, h, 1))
        Hr = H.repeat(1, h, 1).view(batch_size*h*h, 1, h)
        # print(Hr.view(batch_size, h, h, 1, h))

        r = Hl.bmm(Hr).view(batch_size, h, h, h, h) * \
            grad_output.contiguous().view(batch_size, 1, 1, h, h).expand(batch_size, h, h, h, h)
        # print(r.size())
        return -r.sum(-1).sum(-1)
        # print(r)


class GradientBatch(nn.Module):

	def __init__(self):
		super(GradientBatch, self).__init__()
		wx = torch.FloatTensor([-.5, 0, .5]).view(1, 1, 1, 3)
		wy = torch.FloatTensor([[-.5], [0], [.5]]).view(1, 1, 3, 1)
		self.register_buffer('wx', wx)
		self.register_buffer('wy', wy)
		self.padx_func = torch.nn.ReplicationPad2d((1,1,0,0))
		self.pady_func = torch.nn.ReplicationPad2d((0,0,1,1))

	def forward(self, img):
		batch_size, k, h, w = img.size()
		img_ = img.view(batch_size * k, h, w)
		img_ = img_.unsqueeze(1)

		img_padx = self.padx_func(img_)
		img_dx = torch.nn.functional.conv2d(input=img_padx,
											weight=Variable(self.wx),
											stride=1,
											padding=0).squeeze(1)

		img_pady = self.pady_func(img_)
		img_dy = torch.nn.functional.conv2d(input=img_pady,
											weight=Variable(self.wy),
											stride=1,
											padding=0).squeeze(1)

		img_dx = img_dx.view(batch_size, k, h, w)
		img_dy = img_dy.view(batch_size, k, h, w)

		if not isinstance(img, torch.autograd.Variable):
			img_dx = img_dx.data
			img_dy = img_dy.data

		return img_dx, img_dy


def normalize_img_batch(img):
	# per-channel zero-mean and unit-variance of image batch

	# img [in, Tensor N x C x H x W] : batch of images to normalize
	N, C, H, W = img.size()

	# compute per channel mean for batch, subtract from image
	img_vec = img.view(N, C, H * W, 1)
	mean = img_vec.mean(dim=2, keepdim=True)
	img_ = img - mean

	# compute per channel std dev for batch, divide img
	std_dev = img_vec.std(dim=2, keepdim=True)
	img_ = img_ / std_dev

	return img_


def get_input_projection_for_template(input_img, template_image, p):
    # Returns the result of projecting template in input, by warping templates' grid with p,
	# and extracting the sampled pixles from input that match the warped grid.
    _, _, template_h, template_w = template_image.size()   
    return warp_hmg(input_img, p, template_w, template_h)	


def warp_hmg(img, p, template_w=0, template_h=0):
	# perform warping of img batch using homography transform with batch of parameters p
	# img [in, Tensor N x C x H x W] : batch of images to warp
	# p [in, Tensor N x 8 x 1] : batch of warp parameters
	# template_wL the width of the template image
	# template_h: the height of the template image
	#
	# img_warp [out, Tensor N x C x H x W] : batch of warped images
	# mask [out, Tensor N x H x W] : batch of binary masks indicating valid pixels areas

	batch_size, k, h, w = img.size()

	# If received, use the widh and height from the template.
	# This should always be sent when the input and template images are not of the same size.
	if template_h != 0:
		h = template_h
	if template_w != 0:
		w = template_w

	use_variable = isinstance(img, torch.autograd.Variable)

	# Create the regular grid.
	x, y = create_regular_grid(w, h, use_variable)
	#printd(f"Regular grid axis ({w}x{h}):")

	# Create the sampling, warped grid with sub-pixel locations.
	X_warp, Y_warp = create_warped_grid(x, y, batch_size, w, h, p, use_variable)
	#printd(f"Warped grid axis ({w}x{h}):")
	#printd(X_warp)
	#printd(Y_warp)

	img_warp, mask = grid_bilinear_sampling(img, X_warp, Y_warp, h, w)

	return img_warp, mask


def create_regular_grid(w, h, use_variable):
	# Setup the regular grid for the template image, just x and y locations from 0 to w and 0 to h.
	if use_variable:
		x = Variable(torch.arange(w))
		y = Variable(torch.arange(h))
		if USE_CUDA:
			x = x.cuda()
			y = y.cuda()
	else:
		x = torch.arange(w)
		y = torch.arange(h)

	return x, y	


def create_warped_grid(x, y, batch_size, w, h, p, use_variable):
	# Given two arrays of positions x (0 to w) and y (o to h), return the warped grid of sub-pixel locations,
	# based on the parameters in p (times the batch size).
	X, Y = meshgrid(x, y)

	if use_variable:
		if USE_CUDA:
		# create xy matrix, 2 x N
			xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), Variable(torch.ones(1, X.numel(), dtype=X.dtype).cuda())), 0)
		else:
			xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), Variable(torch.ones(1, X.numel(), dtype=X.dtype))), 0)
	else:
		xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), torch.ones(1, X.numel(), dtype=X.dtype)), 0)
	xy = xy.repeat(batch_size, 1, 1)

	H = param_to_H(p)
	xy_warp = H.bmm(xy.float())

	# extract warped X and Y, normalizing the homog coordinates
	X_warp = xy_warp[:,0,:] / xy_warp[:,2,:]
	Y_warp = xy_warp[:,1,:] / xy_warp[:,2,:]
	X_warp = X_warp.view(batch_size,h,w) + (w-1)/2
	Y_warp = Y_warp.view(batch_size,h,w) + (h-1)/2

	return X_warp, Y_warp


def meshgrid(x, y):
	imW = x.size(0)
	imH = y.size(0)

	x = x - x.max()/2
	y = y - y.max()/2

	X = x.unsqueeze(0).repeat(imH, 1)
	Y = y.unsqueeze(1).repeat(1, imW)
	return X, Y


def grid_bilinear_sampling(A, x, y, h_t, w_t):
	batch_size, k, h, w = A.size()
	x_norm = x/((w_t-1)/2) - 1
	y_norm = y/((h_t-1)/2) - 1
	grid = torch.cat((x_norm.view(batch_size, h_t, w_t, 1), y_norm.view(batch_size, h_t, w_t, 1)), 3)
	Q = grid_sample(A, grid, mode='bilinear', align_corners=True)

	if isinstance(A, torch.autograd.Variable):
		if USE_CUDA:
			in_view_mask = Variable(((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h)).type_as(A.data).cuda())
		else:
			in_view_mask = Variable(((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h)).type_as(A.data))
	else:
		in_view_mask = ((x_norm > -1+2/w) & (x_norm < 1-2/w) & (y_norm > -1+2/h) & (y_norm < 1-2/h)).type_as(A)
		Q = Q.data

	return Q.view(batch_size, k, h_t, w_t), in_view_mask

def param_to_H(p):
	# batch parameters to batch homography
	batch_size, _, _ = p.size()

	if isinstance(p, torch.autograd.Variable):
		if USE_CUDA:
			z = Variable(torch.zeros(batch_size, 1, 1).cuda())
		else:
			z = Variable(torch.zeros(batch_size, 1, 1))
	else:
		z = torch.zeros(batch_size, 1, 1)

	p_ = torch.cat((p, z), 1)

	if isinstance(p, torch.autograd.Variable):
		if USE_CUDA:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1).cuda())
		else:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1))
	else:
		I = torch.eye(3,3).repeat(batch_size, 1, 1)

	H = p_.view(batch_size, 3, 3) + I

	return H

def H_to_param(H):
	# batch homography to batch parameters
	batch_size, _, _ = H.size()

	if isinstance(H, torch.autograd.Variable):
		if USE_CUDA:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1).cuda())
		else:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1))
	else:
		I = torch.eye(3,3).repeat(batch_size, 1, 1)


	p = H - I

	p = p.view(batch_size, 9, 1)
	p = p[:, 0:8, :]

	return p


class vgg16Conv(nn.Module):
	def __init__(self, model_path):
		super(vgg16Conv, self).__init__()

		print('Loading pretrained network...',end='')
		vgg16 = torch.load(model_path, map_location=lambda storage, loc: storage)
		print('done')

		self.features = nn.Sequential(
			*(list(vgg16.features.children())[0:15]),
		)

		# freeze conv1, conv2
		for p in self.parameters():
			if p.size()[0] < 256:
				p.requires_grad=False

		'''
	    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (1): ReLU(inplace)
	    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (3): ReLU(inplace)
	    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (6): ReLU(inplace)
	    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (8): ReLU(inplace)
	    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (11): ReLU(inplace)
	    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (13): ReLU(inplace)
	    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (15): ReLU(inplace)
	    (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (18): ReLU(inplace)
	    (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (20): ReLU(inplace)
	    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (22): ReLU(inplace)
	    (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (25): ReLU(inplace)
	    (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (27): ReLU(inplace)
	    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (29): ReLU(inplace)
	    (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    '''

	def forward(self, x):
		# print('CNN stage...',end='')
		x = self.features(x)
		# print('done')
		return x

class custom_net(nn.Module):
	def __init__(self, model_path):
		super(custom_net, self).__init__()

		print('Loading pretrained network...',end='')
		self.custom = torch.load(model_path, map_location=lambda storage, loc: storage)
		print('done')

	def forward(self, x):
		x = self.custom(x)
		return x

class DeepLK(nn.Module):
	def __init__(self, conv_net):
		super(DeepLK, self).__init__()
		self.img_gradient_func = GradientBatch()
		self.conv_func = conv_net
		self.inv_func = InverseBatch

	def forward(self, img, temp, init_param=None, tol=1e-3, max_itr=500, conv_flag=0, ret_itr=False):
		# img: map/reference image.
		# tmp: template image, actual smaller image we are trying to place in the reference one.

		# If flag is enabled, extract features for both images using trained CNN.
		if conv_flag:
			printd("Executing CNN to extract features from image 1...")
			Ft = self.conv_func(temp)
			printd("Finished executing CNN.")

			printd("Executing CNN to extract features from image 2...")
			Fi = self.conv_func(img)
			printd("Finished executing CNN.")
		else:
			Fi = img
			Ft = temp

		printd(f"Fi size: {Fi.size()}")
		printd(f"Ft size: {Ft.size()}")
		batch_size, k, h, w = Ft.size()

		# Compute basic Jacobian matrix from template/small image needed for iterations. This doesn't change between iterations.
		# From the CLKN paper, this is the equivalent of obtaining (J^t*J)^-1*J^t. This will have to be multipled by "r" later to get he delta p dp, in each iteration.
		Ftgrad_x, Ftgrad_y = self.img_gradient_func(Ft)
		dIdp = self.compute_dIdp(Ftgrad_x, Ftgrad_y)
		dIdp_t = dIdp.transpose(1, 2)
		invH = self.inv_func.apply(dIdp_t.bmm(dIdp))
		invH_dIdp = invH.bmm(dIdp_t)

		# User initial param for p, or if that is not provided, an empty tensor.
		p = init_param
		if p is None:
			p = Variable(torch.zeros(batch_size, 8, 1))
			if USE_CUDA:
				p = p.cuda()

		# Initialize the empty change/delta tensor.
		dp = Variable(torch.ones(batch_size, 8, 1))
		if USE_CUDA:
			dp = dp.cuda() # ones so that the norm of each dp is larger than tol for first iteration

		# Iterate to improve the motion params in p until tolerance is reached or we have reached the max number of iterations.
		_, _, template_h, template_w = Ft.size()
		itr = 1
		while (float(dp.norm(p=2,dim=1,keepdim=True).max()) > tol or itr == 1) and (itr <= max_itr):
			# Calculate projected image based on our current motion parameters p. This is done on the map image for some reason.
			Fi_warp, mask = get_input_projection_for_template(Fi, Ft, p)
			#printd(f"Fi_warp size: {Fi_warp.size()}")
			#printd(f"mask size: {mask.size()}")
			#printd(f"number of 1s in mask: {mask.sum()}")
			#printd(mask)

			# Add one dimension to the mask and repeat for that dimension, maybe because the channel (k) dimension is not really used when creating the mask?
			mask.unsqueeze_(1)
			mask = mask.repeat(1, k, 1, 1)

			# TODO: this is appling a mask of "important zones" from the warped sat image to the UAV image. This won't work when the UAV image is smaller. Figure out
			# how to change this to make it work.
			# Multiply the template image by the mask, to get the zones we care about from the template/UAV image?
			#printd(f"Fi size: {Fi.size()}")
			#printd(f"Ft size: {Ft.size()}")
			Ft_mask = Ft.mul(mask)

			# Calculate residual vector r of the error between the warped map image and the masked template/uav image.
			r = Fi_warp - Ft_mask
			r = r.view(batch_size, k * h * w, 1)

			# Calculate the change/delta for the motion parameters p given our current iteration.
			dp_new = invH_dIdp.bmm(r)
			dp_new[:,6:8,0] = 0
			if USE_CUDA:
				dp = (dp.norm(p=2,dim=1,keepdim=True) > tol).type(torch.FloatTensor).cuda() * dp_new
			else:
				dp = (dp.norm(p=2,dim=1,keepdim=True) > tol).type(torch.FloatTensor) * dp_new

			# Update the motion parameters with the delta we obtained, and update iteration number.
			p = p - dp
			itr = itr + 1

		print('finished at iteration ', itr)

		if (ret_itr):
			return p, param_to_H(p), itr
		else:
			return p, param_to_H(p)

	def compute_dIdp(self, Ftgrad_x, Ftgrad_y):

		batch_size, k, h, w = Ftgrad_x.size()

		x = torch.arange(w)
		y = torch.arange(h)

		X, Y = meshgrid(x, y)

		X = X.view(X.numel(), 1)
		Y = Y.view(Y.numel(), 1)

		X = X.repeat(batch_size, k, 1)
		Y = Y.repeat(batch_size, k, 1)

		if USE_CUDA:
			X = Variable(X.cuda())
			Y = Variable(Y.cuda())
		else:
			X = Variable(X)
			Y = Variable(Y)

		Ftgrad_x = Ftgrad_x.view(batch_size, k * h * w, 1)
		Ftgrad_y = Ftgrad_y.view(batch_size, k * h * w, 1)

		dIdp = torch.cat((
			X.mul(Ftgrad_x), 
			Y.mul(Ftgrad_x),
			Ftgrad_x,
			X.mul(Ftgrad_y),
			Y.mul(Ftgrad_y),
			Ftgrad_y,
			-X.mul(X).mul(Ftgrad_x) - X.mul(Y).mul(Ftgrad_y),
			-X.mul(Y).mul(Ftgrad_x) - Y.mul(Y).mul(Ftgrad_y)),2)

		# dIdp size = batch_size x k*h*w x 8
		return dIdp
