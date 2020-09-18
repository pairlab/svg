import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=28, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')
parser.add_argument('--annotations_file', default='', help='annotations file to use for 50salads')
parser.add_argument('--max_dataset_size', default=None, type=int, help='max number of sequences in dataset')
parser.add_argument('--seq_skip', default=1, type=int, help='skip frames between frames in sequence')

opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)


opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.train()
decoder.train()
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for item in train_loader:
            if tmp['opt'].model == '50salads_cond':
                sequence, action = item
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch, action
            else:
                sequence = item
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for item in test_loader:
            if tmp['opt'].model == '50salads_cond':
                sequence, action = item
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch, action
            else:
                sequence = item
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
testing_batch_generator = get_testing_batch()

## guided backprop

"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU, LeakyReLU, Sequential

from vis_module.misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model

        # patch in features
        self.model.features = nn.ModuleList([self.model.c1, self.model.c2, self.model.c3, self.model.c4, self.model.c5])
        
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        # self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            from pdb import set_trace
            set_trace()
            self.gradients = grad_in[0]
            print("grad shape", self.gradients.shape)
        # Register hook to the first layer
        first_vgg_layer = list(self.model.features._modules.items())[0][1][0]
        first_conv_layer = list(first_vgg_layer._modules.items())[0][1][0]
        first_conv_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for _, seqm in self.model.features._modules.items():
            # modules are sequential
            for _, vggm in seqm._modules.items():
                # vgg modules
                for _, seqm2 in vggm._modules.items():
                    # seq modules again
                    for _, m in seqm2._modules.items():
                        # actual modules
                        if isinstance(m, ReLU) or isinstance(m, LeakyReLU):
                            print("registering hook")
                            m.register_backward_hook(relu_backward_hook_function)
                            m.register_forward_hook(relu_forward_hook_function)

    def hook_input(self, input_tensor):
            def hook_function(grad_in):
                self.gradients = grad_in
            input_tensor.register_hook(hook_function)

    def generate_gradients(self, input_image, target_class = None):
        input_image = input_image.clone().detach().requires_grad_(True)
        self.hook_input(input_image)
        # Forward pass
        model_output, intermediate_layers = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop

        if target_class is not None:
            one_hot_output = torch.zeros((1, model_output.size()[-1]))
            one_hot_output[0][target_class] = 1
        else:
            # see activations on entire latent space
            one_hot_output = torch.ones((1, model_output.size()[-1]))
        # Backward pass
        model_output.backward(gradient=one_hot_output.cuda())
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0]

        return gradients_as_arr


if __name__ == '__main__':
    # target_example = 0  # Snake
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(encoder)

    for i in range(len(test_loader)):
        # plot train
        # train_x = next(training_batch_generator)
        # make_gifs(train_x, i, 'train')

        # plot test

        if tmp['opt'].model == '50salads_cond':
            test_x, action = next(testing_batch_generator)
        else:
            test_x = next(testing_batch_generator)
            action = None

        # from pdb import set_trace
        # set_trace()

        # take the first image in each sequence of batch
        for b in range(opt.batch_size):
            prep_img = test_x[0][b]
            prep_img = torch.unsqueeze(prep_img, 0)

            file_name_to_export = 'grad_{}'.format(b)

            # Get gradients
            guided_grads = GBP.generate_gradients(prep_img)
            inp = save_gradient_images(torch.squeeze(prep_img, 0).cpu().data.numpy(), file_name_to_export + '_input_image', opt.log_dir)
            # Save colored gradients
            grads_col = save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color', opt.log_dir)
            # Convert to grayscale
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            # Save grayscale gradients
            grads_grey = save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray', opt.log_dir)
            # Positive and negative saliency maps
            pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
            pos_sal_img = save_gradient_images(pos_sal, file_name_to_export + '_pos_sal', opt.log_dir)
            neg_sal_img = save_gradient_images(neg_sal, file_name_to_export + '_neg_sal', opt.log_dir)

            # make grayscale 3 channels.
            grads_grey = np.concatenate((grads_grey, grads_grey, grads_grey), 0)
            # input, coloured gradients, greyscale gradients, positive sal, negative sal
            all_imgs = np.hstack((inp, grads_col, grads_grey, pos_sal_img, neg_sal_img))
            save_gradient_images(all_imgs, file_name_to_export + '_ALL', opt.log_dir)

            print('Guided backprop completed for {}'.format(b))
