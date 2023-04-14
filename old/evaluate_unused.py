
def random_param_generator():
	# create random ground truth warp parameters in the specified ranges

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

	return p_gt



def static_data_generator(batch_size):
	# similar to data_generator, except with static warp parameters (easier for testing)

	FOLDERPATH = DATAPATH + FOLDER
	FOLDERPATH = FOLDERPATH + 'images/'
	images_dir = glob.glob(FOLDERPATH + '*.png')
	random.shuffle(images_dir)

	img = Image.open(images_dir[0])
	template = Image.open(images_dir[1])

	in_W, in_H = img.size

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
		seg_sz = 200
		seg_sz_pad = round(seg_sz + seg_sz * 2 * warp_pad)

		loc_x = 40
		loc_y = 40

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

		scale = 1.2
		angle = 10
		projective_x = 0
		projective_y = 0
		translation_x = -5
		translation_y = 10

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

		img_batch[i, :, :, :] = img_seg_w[0:3,:,:]
		template_batch[i, :, :, :] = template_seg[0:3,:,:]

		param_batch[i, :, :] = p_gt[0, :, :].data

	return img_batch, template_batch, param_batch

