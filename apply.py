import argparse

import torch
from torch.autograd import Variable

import DeepLKBatch as dlk


# rotation range (-angle_range, angle_range)
angle_range = 15 # degrees

# projective variables (p7, p8)
projective_range = 0

# translation (p3, p6)
translation_range = 10 # pixels

# possible segment sizes
lower_sz = 200 # pixels, square
upper_sz = 220

# amount to pad when cropping segment, as ratio of size, on all 4 sides
warp_pad = 0.4

# normalized size of all training pairs
training_sz = 175
training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)

USE_CUDA = torch.cuda.is_available()
            

def test():
    dlk_trained = dlk.DeepLK(dlk.custom_net(MODEL_PATH))
    if USE_CUDA:
        dlk_trained = dlk_trained.cuda()

    img_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz))
    template_test_data = Variable(torch.zeros(test_rounds_num, 3, training_sz, training_sz))
    param_test_data = Variable(torch.zeros(test_rounds_num, 8, 1))
    if USE_CUDA:
        img_test_data = img_test_data.cuda()
        template_test_data = template_test_data.cuda()
        param_test_data = param_test_data.cuda()

    # TODO: load this images
    #img_batch, template_batch, param_batch = data_generator(rounds_per_pair)
    
    img_test_data[batch_index:batch_index + rounds_per_pair, :, :, :] = img_batch
    template_test_data[batch_index:batch_index + rounds_per_pair, :, :, :] = template_batch
    param_test_data[batch_index:batch_index + rounds_per_pair, :, :] = param_batch

    img_batch_unnorm = img_test_data[i, :, :, :].unsqueeze(0)
    template_batch_unnorm = template_test_data[i, :, :, :].unsqueeze(0)
    param_batch = param_test_data[i, :, :].unsqueeze(0)

    img_batch = dlk.normalize_img_batch(img_batch_unnorm)
    template_batch = dlk.normalize_img_batch(template_batch_unnorm)

    trained_param, _  = dlk_trained(img_batch, template_batch, tol=1e-3, max_itr=70, conv_flag=1)

    # STEPS
    # 0. LATER: Add additional steps to get an image close to the picture from the map, given assumptions on where we are.
    # 1. Obtain 2 similar images (created manually, externally from this app), as inputs
    # 2. Load images properly (load them, normalize them, etc).
    # 3. Run the dlk_trained on these two images (it receives two batches, but in this case each batch will be of 1)
    # 4. Check out the params and homography matrix from dlk
    # 5. Use matrix to apply it to one image, and somehow store the modified image to see resutlts?
    # 6. LATER: convert to GPS coordinates.
