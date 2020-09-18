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
opt.action_space = tmp['opt'].action_space

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
            sequence, action = item
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch, action
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for item in test_loader:
            sequence, action = item
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch, action
testing_batch_generator = get_testing_batch()

# --------- eval funtions ------------------------------------

def make_gifs(x, idx, name, action, action_indx):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]

    action = action.cuda()

    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t, action], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t, action], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    mse = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []

    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t, action], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t, action], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        mse[:, s, :], ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)
    
    action_gen = []

    for a in range(opt.action_space):
        new_action = torch.zeros_like(action)
        new_action[a] = 1

        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        action_gen.append([])
        action_gen[a].append(x_in)

        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t, new_action], 1))
                x_in = x[i]
                action_gen[a].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t, new_action], 1)).detach()
                x_in = decoder([h, skip]).detach()
                action_gen[a].append(x_in)

    progress.finish()
    utils.clear_progressbar()

    avg_best_ssim = np.zeros((opt.n_future,))
    avg_best_mse = np.zeros((opt.n_future,))
    avg_best_psnr = np.zeros((opt.n_future,))

    ###### ssim ######
    for i in range(opt.batch_size):
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(opt.n_eval):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append("Ground\ntruth ({})".format(action_indx[i]))
            #posterior 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM: \n {:.5f}'.format(mean_ssim[sidx]))
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))
            
            for a in range(opt.action_space):
                gifs[t].append(add_border(action_gen[a][t][i], color))
                text[t].append('Action\nsample %d' % (a))

        fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)

        best_seq_ssim = ssim[i][ordered[-1]]
        avg_best_ssim += best_seq_ssim / opt.batch_size

        mean_mse = np.mean(mse[i], 1)
        ordered = np.argsort(mean_mse)
        best_seq_mse = mse[i][ordered[-1]]
        avg_best_mse += best_seq_mse / opt.batch_size

        mean_psnr = np.mean(psnr[i], 1)
        ordered = np.argsort(mean_psnr)
        best_seq_psnr = psnr[i][ordered[-1]]
        avg_best_psnr += best_seq_psnr / opt.batch_size
    
    return avg_best_mse, avg_best_ssim, avg_best_psnr

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

# for i in range(0, opt.N, opt.batch_size):
# Just iterate entire test set.
print("iterating test set of size ", len(test_loader))

avg_best_mse = np.zeros((opt.n_future,))
avg_best_ssim = np.zeros((opt.n_future,))
avg_best_psnr = np.zeros((opt.n_future,))

for i in range(len(test_loader)):
    # plot train
    # train_x = next(training_batch_generator)
    # make_gifs(train_x, i, 'train')

    # plot test
    test_x, action = next(testing_batch_generator)

    action_indx = []
    for j in range(len(action)):
        for a in range(len(action[j])):
            if action[j][a] == 1:
                action_indx.append(a)
                continue

    abm, abss, abp = make_gifs(test_x, i, 'test', action, action_indx)

    avg_best_mse += abm/len(test_loader)
    avg_best_ssim += abss/len(test_loader)
    avg_best_psnr += abp/len(test_loader)

    print("finished iter {}. avg_best_mse: {} avg_best_ssim: {} avg_best_psnr: {}".format(i, abm, abss, abp))

    np.savetxt(os.path.join(opt.log_dir, 'mse_ssim_psnr.txt'), np.vstack((avg_best_mse, \
        avg_best_ssim, avg_best_psnr)), delimiter = ",")


print("finished generation. overall avg_best_mse: {} avg_best_ssim: {} avg_best_psnr: {}".format(avg_best_mse, avg_best_ssim, avg_best_psnr))
print("model path", opt.model_path)