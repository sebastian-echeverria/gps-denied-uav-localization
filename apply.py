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

###--- TRAINING/TESTING PARAMETERS
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("MODE")
    parser.add_argument("FOLDER_NAME")
    parser.add_argument("DATAPATH")
    parser.add_argument("MODEL_PATH")
    parser.add_argument("VGG_MODEL_PATH")
    parser.add_argument("-t","--TEST_DATA_SAVE_PATH")

    args = parser.parse_args()

    MODE = args.MODE
    FOLDER_NAME = args.FOLDER_NAME
    FOLDER = FOLDER_NAME + '/'
    DATAPATH = args.DATAPATH
    MODEL_PATH = args.MODEL_PATH
    VGG_MODEL_PATH = args.VGG_MODEL_PATH

    if MODE == 'test':
        if args.TEST_DATA_SAVE_PATH == None:
            exit('Must supply TEST_DATA_SAVE_PATH argument in test mode')
        else:
            TEST_DATA_SAVE_PATH = args.TEST_DATA_SAVE_PATH
            

def test():
    dlk_trained = dlk.DeepLK(dlk.custom_net(MODEL_PATH))
    if USE_CUDA:
        dlk_trained = dlk_trained.cuda()

    testbatch_sz = 1 # keep as 1 in order to compute corner error accurately
    test_rounds_num = 50
    rounds_per_pair = 50

    print('Testing...')
    print('TEST DATA SAVE PATH: ', TEST_DATA_SAVE_PATH)
    print('DATAPATH: ',DATAPATH)
    print('FOLDER: ', FOLDER)
    print('MODEL PATH: ', MODEL_PATH)
    print('USE CUDA: ', USE_CUDA)
    print('angle_range: ', angle_range)
    print('projective_range: ', projective_range)
    print('translation_range: ', translation_range)
    print('lower_sz: ', lower_sz)
    print('upper_sz: ', upper_sz)
    print('warp_pad: ', warp_pad)
    print('test batch size: ', testbatch_sz, ' number of test round: ', test_rounds_num, ' rounds per pair: ', rounds_per_pair)

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

