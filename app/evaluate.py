import sys
import os
import time
import argparse
import gc
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import DeepLKBatch as dlk
import image_generator

# suppress endless SourceChangeWarning messages from pytorch
import warnings
warnings.filterwarnings("ignore")

# USAGE:
# python3 evaluate.py MODE SAT_PATH MODEL_PATH VGG_MODEL_PATH [--TEST_DATA_SAVE_PATH]

# TRAIN:
# python3 evaluate.py train ../sat_data/woodbridge/ trained_model_output.pth ../models/vgg16_model.pth

# TEST:
# python3 evaluate.py test ../sat_data/woodbridge/ ../models/conv_02_17_18_1833.pth ../models/vgg16_model.pth -t test_out.txt

###--- TRAINING/TESTING PARAMETERS

# amount to pad when cropping segment, as ratio of size, on all 4 sides
warp_pad = 0.4

# normalized size of all training pairs
training_sz = 175
training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)

USE_CUDA = torch.cuda.is_available()

###---

def corner_loss(p, p_gt):
	# p [in, torch tensor] : batch of regressed warp parameters
	# p_gt [in, torch tensor] : batch of gt warp parameters
	# loss [out, float] : sum of corner loss over minibatch

	batch_size, _, _ = p.size()

	# compute corner loss
	H_p = dlk.param_to_H(p)
	H_gt = dlk.param_to_H(p_gt)

	if USE_CUDA:
		corners = Variable(torch.Tensor([[-training_sz_pad/2, training_sz_pad/2, training_sz_pad/2, -training_sz_pad/2],
								[-training_sz_pad/2, -training_sz_pad/2, training_sz_pad/2, training_sz_pad/2],
								[1, 1, 1, 1]]).cuda())
	else:
		corners = Variable(torch.Tensor([[-training_sz_pad/2, training_sz_pad/2, training_sz_pad/2, -training_sz_pad/2],
								[-training_sz_pad/2, -training_sz_pad/2, training_sz_pad/2, training_sz_pad/2],
								[1, 1, 1, 1]]))

	corners = corners.repeat(batch_size, 1, 1)

	corners_w_p = H_p.bmm(corners)
	corners_w_gt = H_gt.bmm(corners)

	corners_w_p = corners_w_p[:, 0:2, :] / corners_w_p[:, 2:3, :]
	corners_w_gt = corners_w_gt[:, 0:2, :] / corners_w_gt[:, 2:3, :]

	loss = ((corners_w_p - corners_w_gt) ** 2).sum()

	return loss

def test(args):
	if USE_CUDA:
		dlk_vgg16 = dlk.DeepLK(dlk.vgg16Conv(args.MODEL_PATH)).cuda()
		dlk_trained = dlk.DeepLK(dlk.custom_net(args.MODEL_PATH)).cuda()
	else:
		dlk_vgg16 = dlk.DeepLK(dlk.vgg16Conv(args.MODEL_PATH))
		dlk_trained = dlk.DeepLK(dlk.custom_net(args.MODEL_PATH))

	testbatch_sz = 1 # keep as 1 in order to compute corner error accurately
	test_rounds_num = 50
	rounds_per_pair = 50

	test_results = np.zeros((test_rounds_num, 5), dtype=float)

	print('Testing...')
	print('TEST DATA SAVE PATH: ', args.TEST_DATA_SAVE_PATH)
	print('SAT_PATH: ', args.SAT_PATH)
	print('MODEL PATH: ', args.MODEL_PATH)
	print('USE CUDA: ', USE_CUDA)
	print('warp_pad: ', warp_pad)
	print('test batch size: ', testbatch_sz, ' number of test round: ', test_rounds_num, ' rounds per pair: ', rounds_per_pair)

	if USE_CUDA:
		img_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz)).cuda()
		template_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz)).cuda()
		param_test_data = Variable(torch.zeros(test_rounds_num, 8, 1)).cuda()
	else:
		img_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz))
		template_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz))
		param_test_data = Variable(torch.zeros(test_rounds_num, 8, 1))

	for i in range(round(test_rounds_num / rounds_per_pair)):
		print('gathering data...', i+1, ' / ', test_rounds_num / rounds_per_pair)
		batch_index = i * rounds_per_pair

		img_batch, template_batch, param_batch = image_generator.generate_image_pairs(args.SAT_PATH, rounds_per_pair, training_sz, training_sz_pad, warp_pad)

		img_test_data[batch_index:batch_index + rounds_per_pair, :, :, :] = img_batch
		template_test_data[batch_index:batch_index + rounds_per_pair, :, :, :] = template_batch
		param_test_data[batch_index:batch_index + rounds_per_pair, :, :] = param_batch

		sys.stdout.flush()

	for i in range(test_rounds_num):
		img_batch_unnorm = img_test_data[i, :, :, :].unsqueeze(0)
		template_batch_unnorm = template_test_data[i, :, :, :].unsqueeze(0)
		param_batch = param_test_data[i, :, :].unsqueeze(0)

		img_batch = dlk.normalize_img_batch(img_batch_unnorm)
		template_batch = dlk.normalize_img_batch(template_batch_unnorm)

		img_batch_coarse = nn.AvgPool2d(4)(img_batch)
		template_batch_coarse = nn.AvgPool2d(4)(template_batch)

		vgg_param, _ = dlk_vgg16(img_batch, template_batch, tol=1e-3, max_itr=70, conv_flag=1)
		trained_param, _  = dlk_trained(img_batch, template_batch, tol=1e-3, max_itr=70, conv_flag=1)
		coarse_param, _ = dlk_vgg16(img_batch_coarse, template_batch_coarse, tol=1e-3, max_itr=70, conv_flag=0)

		vgg_loss = corner_loss(vgg_param, param_batch)

		if USE_CUDA:
			no_op_loss = corner_loss(Variable(torch.zeros(testbatch_sz, 8, 1)).cuda(), param_batch)
		else:
			no_op_loss = corner_loss(Variable(torch.zeros(testbatch_sz, 8, 1)), param_batch)
		
		trained_loss = corner_loss(trained_param, param_batch)
		coarse_loss = corner_loss(coarse_param, param_batch)

		# srh_param = srh.get_param(img_batch_unnorm, template_batch_unnorm, training_sz)
		# no_op_loss = corner_loss(Variable(torch.zeros(testbatch_sz, 8, 1)), param_batch)
		# srh_loss = corner_loss(srh_param, param_batch)
		
		test_results[i, 0] = vgg_loss
		test_results[i, 1] = trained_loss
		test_results[i, 2] = coarse_loss
		test_results[i, 3] = no_op_loss
		# test_results[i, 4] = srh_loss

		print('test: ', i, 
			' vgg16 loss: ', round(sqrt(float(vgg_loss)/4),2), 
			' trained loss: ', round(sqrt(float(trained_loss)/4),2), 
			' coarse pix loss: ', round(sqrt(float(coarse_loss)/4),2), 
			' no-op loss: ', round(sqrt(float(no_op_loss)/4),2))
			# ' srh loss: ', round(sqrt(float(srh_loss)/4),2))

		sys.stdout.flush()

		#### --- Visualize Testing

		# warped_back_custom, _ = dlk.warp_hmg(img_batch_unnorm, trained_param)
		# warped_back_srh, _ = dlk.warp_hmg(img_batch_unnorm, srh_param)
		# warped_back_vgg, _ = dlk.warp_hmg(img_batch_unnorm, vgg_param)
		# warped_back_iclk, _ = dlk.warp_hmg(img_batch_unnorm, coarse_param)

		# transforms.ToPILImage()(img_batch_unnorm[0,:,:,:].data).show()
		# time.sleep(0.25)
		# transforms.ToPILImage()(template_batch_unnorm[0,:,:,:].data).show()
		# time.sleep(0.25)
		# transforms.ToPILImage()(warped_back_custom[0,:,:,:].data).show()
		# time.sleep(0.25)
		# transforms.ToPILImage()(warped_back_srh[0,:,:,:].data).show()
		# time.sleep(0.25)
		# transforms.ToPILImage()(warped_back_vgg[0,:,:,:].data).show()
		# time.sleep(0.25)
		# transforms.ToPILImage()(warped_back_iclk[0,:,:,:].data).show()

		# pdb.set_trace()

		#### ---

	np.savetxt(args.TEST_DATA_SAVE_PATH, test_results, delimiter=',')



def train(args):
	if USE_CUDA:
		dlk_net = dlk.DeepLK(dlk.vgg16Conv(args.MODEL_PATH)).cuda()
	else:
		dlk_net = dlk.DeepLK(dlk.vgg16Conv(args.MODEL_PATH))

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, dlk_net.conv_func.parameters()), lr=0.0001)

	best_valid_loss = 0

	minibatch_sz = 10
	num_minibatch = 25000
	valid_batch_sz = 10
	valid_num_generator = 50

	print('Training...')
	print('SAT_PATH: ', args.SAT_PATH)
	print('MODEL_PATH: ', args.MODEL_PATH)
	print('VGG MODEL PATH', args.MODEL_PATH)
	print('USE CUDA: ', USE_CUDA)
	print('warp_pad: ', warp_pad)
	print('minibatch size: ', minibatch_sz, ' number of minibatches: ', num_minibatch)

	if USE_CUDA:
		img_train_data = Variable(torch.zeros(num_minibatch, 3, training_sz, training_sz)).cuda()
		template_train_data = Variable(torch.zeros(num_minibatch, 3, training_sz, training_sz)).cuda()
		param_train_data = Variable(torch.zeros(num_minibatch, 8, 1)).cuda()
	else:
		img_train_data = Variable(torch.zeros(num_minibatch, 3, training_sz, training_sz))
		template_train_data = Variable(torch.zeros(num_minibatch, 3, training_sz, training_sz))
		param_train_data = Variable(torch.zeros(num_minibatch, 8, 1))

	for i in range(round(num_minibatch / minibatch_sz)):
		print('gathering training data...', i+1, ' / ', num_minibatch / minibatch_sz)
		batch_index = i * minibatch_sz

		img_batch, template_batch, param_batch = image_generator.generate_image_pairs(args.SAT_PATH, minibatch_sz, training_sz, training_sz_pad, warp_pad)

		img_train_data[batch_index:batch_index + minibatch_sz, :, :, :] = img_batch
		template_train_data[batch_index:batch_index + minibatch_sz, :, :, :] = template_batch
		param_train_data[batch_index:batch_index + minibatch_sz, :, :] = param_batch
		sys.stdout.flush()

	if USE_CUDA:
		valid_img_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz)).cuda()
		valid_template_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz)).cuda()
		valid_param_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 8, 1)).cuda()
	else:
		valid_img_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz))
		valid_template_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz))
		valid_param_batch = Variable(torch.zeros(valid_batch_sz * valid_num_generator, 8, 1))

	for i in range(valid_num_generator):
		print('gathering validation data...', i+1, ' / ', valid_num_generator)
		valid_img_batch[i * valid_batch_sz: i * valid_batch_sz + valid_batch_sz,:,:,:], valid_template_batch[i * valid_batch_sz: i * valid_batch_sz + valid_batch_sz,:,:,:], valid_param_batch[i * valid_batch_sz: i * valid_batch_sz + valid_batch_sz,:,:] = image_generator(valid_batch_sz)


	for i in range(num_minibatch):
		start = time.time()
		optimizer.zero_grad()

		img_batch_unnorm = img_train_data[i, :, :, :].unsqueeze(0)
		template_batch_unnorm = template_train_data[i, :, :, :].unsqueeze(0)
		training_param_batch = param_train_data[i, :, :].unsqueeze(0)

		training_img_batch = dlk.normalize_img_batch(img_batch_unnorm)
		training_template_batch = dlk.normalize_img_batch(template_batch_unnorm)

		# 	forward pass of training minibatch through dlk
		dlk_param_batch, _ = dlk_net(training_img_batch, training_template_batch, tol=1e-3, max_itr=1, conv_flag=1)

		loss = corner_loss(dlk_param_batch, training_param_batch)

		loss.backward()

		optimizer.step()

		dlk_valid_param_batch, _ = dlk_net(valid_img_batch, valid_template_batch, tol=1e-3, max_itr=1, conv_flag=1)

		valid_loss = corner_loss(dlk_valid_param_batch, valid_param_batch)

		print('mb: ', i+1, ' training loss: ', float(loss), ' validation loss: ', float(valid_loss), end='')

		if (i == 0) or (float(valid_loss) < float(best_valid_loss)):
			best_valid_loss = valid_loss
			torch.save(dlk_net.conv_func, args.MODEL_PATH)
			print(' best validation loss: ', float(best_valid_loss), ' (saving)')
		else:
			print(' best validation loss: ', float(best_valid_loss))

		gc.collect()
		sys.stdout.flush()

		end = time.time()

		# print('elapsed: ', end-start)


if __name__ == "__main__":
	print('PID: ', os.getpid())

	parser = argparse.ArgumentParser()
	parser.add_argument("MODE")
	parser.add_argument("SAT_PATH")
	parser.add_argument("MODEL_PATH")
	parser.add_argument("VGG_MODEL_PATH")
	parser.add_argument("-t","--TEST_DATA_SAVE_PATH")

	args = parser.parse_args()

	if args.MODE == 'test':
		if args.TEST_DATA_SAVE_PATH == None:
			exit('Must supply TEST_DATA_SAVE_PATH argument in test mode')
		test(args)
	else:
		train(args)
