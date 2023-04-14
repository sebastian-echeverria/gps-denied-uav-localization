
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

def main():
	sz = 200
	xy = [0, 0]
	sm_factor = 8

	sz_sm = int(sz/sm_factor)

	# conv_flag = int(argv[3])

	preprocess = transforms.Compose([
		transforms.ToTensor(),
	])

	img1 = Image.open(argv[1]).crop((xy[0], xy[1], xy[0]+sz, xy[1]+sz))
	img1_coarse = Variable(preprocess(img1.resize((sz_sm, sz_sm))))
	img1 = Variable(preprocess(img1))

	img2 = Image.open(argv[2]).crop((xy[0], xy[1], xy[0]+sz, xy[1]+sz))
	img2_coarse = Variable(preprocess(img2.resize((sz_sm, sz_sm))))
	img2 = Variable(preprocess(img2)) #*Variable(0.2*torch.rand(3,200,200)-1)

	transforms.ToPILImage()(img1.data).show()
	# transforms.ToPILImage()(img2.data).show()

	scale = 1.6
	angle = 15
	projective_x = 0
	projective_y = 0
	translation_x = 0
	translation_y = 0

	rad_ang = angle / 180 * pi

	p = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
							   -sin(rad_ang),
							   translation_x,
							   sin(rad_ang),
							   scale + cos(rad_ang) - 2,
							   translation_y,
							   projective_x, 
							   projective_y]))
	p = p.view(8,1)
	pt = p.repeat(5,1,1)

	# p = Variable(torch.Tensor([0.4, 0, 0, 0, 0, 0, 0, 0]))
	# p = p.view(8,1)
	# pt = torch.cat((p.repeat(10,1,1), pt), 0)

	# print(p)

	dlk = DeepLK()

	img1 = img1.repeat(5,1,1,1)
	img2 = img2.repeat(5,1,1,1)
	img1_coarse = img1_coarse.repeat(5,1,1,1)
	img2_coarse = img2_coarse.repeat(5,1,1,1)

	wimg2, _ = warp_hmg(img2, H_to_param(dlk.inv_func.apply(param_to_H(pt))))

	wimg2_coarse, _ = warp_hmg(img2_coarse, H_to_param(dlk.inv_func.apply(param_to_H(pt))))

	transforms.ToPILImage()(wimg2[0,:,:,:].data).show()

	img1_n = normalize_img_batch(img1)
	wimg2_n = normalize_img_batch(wimg2)

	img1_coarse_n = normalize_img_batch(img1_coarse)
	wimg2_coarse_n = normalize_img_batch(wimg2_coarse)

	start = time.time()
	print('start conv...')
	p_lk_conv, H_conv = dlk(wimg2_n, img1_n, tol=1e-4, max_itr=200, conv_flag=1)
	print('conv time: ', time.time() - start)

	start = time.time()
	print('start raw...')
	p_lk, H = dlk(wimg2_coarse_n, img1_coarse_n, tol=1e-4, max_itr=200, conv_flag=0)
	print('raw time: ', time.time() - start)

	print((p_lk_conv[0,:,:]-pt[0,:,:]).norm())
	print((p_lk[0,:,:]-pt[0,:,:]).norm())
	print(H_conv)
	print(H)

	warped_back_conv, _ = warp_hmg(wimg2, p_lk_conv)
	warped_back_lk, _ = warp_hmg(wimg2, p_lk) 

	transforms.ToPILImage()(warped_back_conv[0,:,:,:].data).show()
	transforms.ToPILImage()(warped_back_lk[0,:,:,:].data).show()

	#conv_loss = evaluate.corner_loss(p_lk_conv, pt)
	#lk_loss = evaluate.corner_loss(p_lk, pt)

	pdb.set_trace()

if __name__ == "__main__":
	main()
