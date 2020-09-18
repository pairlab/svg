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

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--annotations_file', default='', help='annotations file to use for 50salads')
parser.add_argument('--max_dataset_size', default=None, type=int, help='max number of sequences in dataset')
parser.add_argument('--seq_skip', default=1, type=int, help='skip frames between frames in sequence')
parser.add_argument('--action_method', default='', help='method used to inject action information.')
# This is one of concat|add|ada_in
parser.add_argument('--action_repr', default='', help='representation for action information.')
# This is one of one_hot|fc
parser.add_argument('--action_skips', default='', help='additional skip connections for action embedding')
# One or more of prior_enc|pred_enc
parser.add_argument('--pe_dim', type=int, default=32, help='dimensionality of prior encoder. 0 will not use one.')
parser.add_argument('--action_scale', type=int, default=2, help='dim of action embedding relative to pe_dim')
parser.add_argument('--valid_actions', default='', help='actions to include in datasets. blank includes all.')
parser.add_argument('--det_lstm', type=int, default=0, help='1 for deterministic lstm')

opt = parser.parse_args()

if opt.model_dir != '' and os.path.isfile(os.path.join(opt.model_dir, 'model_latest.pth')) :
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model_latest.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    # opt.log_dir = '%s/continued' % opt.log_dir
    start_epoch = opt.start_epoch
else:
    name = 'type=lp-model=%s%dx%d-n_past=%d-n_future=%d-z_dim=%d-g_dim=%d-pe_dim=%d-action_scale=%d-beta=%.7f%s-batch_size=%d-action_method=%s-action_repr=%s-valid_actions=%s\
-prior_rnn_lay=%d-det_lstm=%d' %\
     (opt.model, opt.image_width, opt.image_width, opt.n_past, \
     opt.n_future, opt.z_dim, opt.g_dim, opt.pe_dim, opt.action_scale, opt.beta, opt.name, \
     opt.batch_size, opt.action_method, opt.action_repr, opt.valid_actions, opt.prior_rnn_layers, opt.det_lstm)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    opt.start_epoch = 0
    start_epoch = opt.start_epoch

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# --- hard-coded options ---#
if opt.dataset == 'utd_actions':
    opt.action_space = 27
elif opt.dataset == '50salads_cond':
    opt.action_space = 4
else:
    opt.action_space = 0

if opt.det_lstm == 1:
    opt.det_lstm = True
else:
    opt.det_lstm = False


# ---------------- load the models  ----------------

print(opt)
f = open(os.path.join(opt.log_dir, 'opt.txt'), 'w')
f.write(str(opt))
f.close()

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)


import models.lstm as lstm_models

if opt.model_dir != '' and os.path.isfile(os.path.join(opt.model_dir, 'model_latest.pth')):
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:

    z_dim = opt.z_dim
    if opt.action_method == 'skip_posterior':
        if opt.action_repr == 'one_hot':
            z_dim = opt.action_space
        elif opt.action_repr == 'fc':
            z_dim = opt.g_dim * opt.action_scale

    if opt.pe_dim == 0:
        frame_predictor_input = opt.g_dim + z_dim
    else:
        frame_predictor_input = opt.pe_dim + z_dim
    frame_predictor = lstm_models.lstm(frame_predictor_input, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size, opt.det_lstm)
    
    prior_input_size = opt.pe_dim if opt.pe_dim != 0 else opt.g_dim
    if opt.action_method == 'concat':
        if opt.action_repr == 'one_hot':
            prior_input_size += opt.action_space
        elif opt.action_repr == 'fc':
            prior_input_size += opt.pe_dim * opt.action_scale if opt.pe_dim != 0 else opt.g_dim * opt.action_scale

    prior = lstm_models.gaussian_lstm(prior_input_size, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size, opt.det_lstm)
    
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
       
if opt.model_dir != '' and os.path.isfile(os.path.join(opt.model_dir, 'model_latest.pth')):
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']

    if "prior_encoder" in saved_model:
        prior_encoder = saved_model['prior_encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

    if opt.pe_dim != 0:
        prior_encoder = model.encoder(opt.pe_dim, opt.channels)
        prior_encoder.apply(utils.init_weights)

if opt.action_repr == 'fc':
    from models.fc import fc

    output_size = opt.pe_dim * opt.action_scale if opt.pe_dim != 0 else opt.g_dim * opt.action_scale

    action_fc = fc(opt.action_space, output_size, output_size, 8) # set output size to same as encoder for now
    action_fc_optimizer = opt.optimizer(action_fc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    action_fc.cuda()

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.pe_dim != 0:
    prior_encoder_optimizer = opt.optimizer(prior_encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
# mse_criterion = nn.L1Loss() # Try L1 loss instead

def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    # 
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()

if opt.pe_dim != 0:
    prior_encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
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

# --------- plotting funtions ------------------------------------
def plot(x, one_hot_action, epoch, metrics_only = False, not_gt_action = False):
    nsample = 20 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    ssim = np.zeros((opt.batch_size, nsample, opt.n_eval - opt.n_past))
    mses = np.zeros((opt.batch_size, nsample, opt.n_eval - opt.n_past))
    psnr = np.zeros((opt.batch_size, nsample,opt.n_eval - opt.n_past))

    action_indx = []
    for j in range(len(one_hot_action)):
        for a in range(len(one_hot_action[j])):
            if one_hot_action[j][a] == 1:
                action_indx.append(a)
                continue

    progress = progressbar.ProgressBar(max_value=nsample).start()

    if opt.action_repr == 'fc':
        action = action_fc(one_hot_action)
    elif opt.action_repr == 'one_hot':
        action = one_hot_action

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]

        # for ssim and psnr metrics
        gen_seq_cpu = []
        gt_seq_cpu = []

        for i in range(1, opt.n_eval):
            if opt.pe_dim != 0:
                h = prior_encoder(x_in)
            else:
                h = encoder(x_in)

            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])
                h_target = h_target[0].detach()

                if opt.action_method == 'skip_posterior':
                    z_t = action
                else:
                    z_t, _, _ = posterior(h_target)

                if opt.action_method == 'concat':
                    prior(torch.cat([h, action], 1))
                elif opt.action_method == 'add':
                    # must be FC with matching output size to h.                        
                    h_and_action = h + action        
                    prior(h_and_action)
                elif opt.action_method == 'no_action':
                    prior(h)
                elif opt.action_method == 'skip_posterior':
                    pass
                else:
                    raise ValueError('Invalid action method')

                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                if opt.action_method == 'concat':
                    z_t, _, _ = prior(torch.cat([h, action], 1))
                elif opt.action_method == 'add':
                    # must be FC with matching output size to h.                        
                    h_and_action = h + action        
                    z_t, _, _ = prior(h_and_action)
                elif opt.action_method == 'no_action':
                    z_t, _, _ = prior(h)
                elif opt.action_method == 'skip_posterior':
                    z_t = action
                else:
                    raise ValueError('Invalid action method')

                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)
                gen_seq_cpu.append(x_in.data.cpu().numpy())
                gt_seq_cpu.append(x[i].data.cpu().numpy())

        mses[:, s, :], ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq_cpu, gen_seq_cpu)

    if metrics_only == False:
        to_plot = []
        gifs = [ [] for t in range(opt.n_eval) ]
        nrow = min(opt.batch_size, 10)
        for i in range(nrow):
            # ground truth sequence
            row = [] 
            for t in range(opt.n_eval):
                row.append(gt_seq[t][i])
            to_plot.append(row)

            # best sequence
            min_mse = float('inf')
            for s in range(nsample):
                mse = 0
                for t in range(opt.n_eval):
                    mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
                if mse < min_mse:
                    min_mse = mse
                    min_idx = s

            s_list = [min_idx, 
                    np.random.randint(nsample), 
                    np.random.randint(nsample), 
                    np.random.randint(nsample), 
                    np.random.randint(nsample)]
            for ss in range(len(s_list)):
                s = s_list[ss]
                row = []
                for t in range(opt.n_eval):
                    row.append(gen_seq[s][t][i]) 
                to_plot.append(row)
            for t in range(opt.n_eval):
                row = []
                row.append(gt_seq[t][i])
                for ss in range(len(s_list)):
                    s = s_list[ss]
                    row.append(gen_seq[s][t][i])
                gifs[t].append(row)

        if not_gt_action:
            fname = '%s/gen/%d/action_sample_%d.png' % (opt.log_dir, epoch, action_indx[0]) 
        else:
            fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch) 
        utils.save_tensors_image(fname, to_plot)

        if not_gt_action:
            fname = '%s/gen/%d/action_sample_%d.gif' % (opt.log_dir, epoch, action_indx[0]) 
        else:
            fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
        utils.save_gif(fname, gifs)

    avg_best_ssim = np.zeros((opt.n_eval - opt.n_past,))
    avg_best_mse = np.zeros((opt.n_eval - opt.n_past,))
    avg_best_psnr = np.zeros((opt.n_eval - opt.n_past,))

    for i in range(opt.batch_size):
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        best_seq_ssim = ssim[i][ordered[-1]]
        avg_best_ssim += best_seq_ssim / opt.batch_size

        mean_mse = np.mean(mses[i], 1)
        ordered = np.argsort(mean_mse)
        best_seq_mse = mses[i][ordered[-1]]
        avg_best_mse += best_seq_mse / opt.batch_size

        mean_psnr = np.mean(psnr[i], 1)
        ordered = np.argsort(mean_psnr)
        best_seq_psnr = psnr[i][ordered[-1]]
        avg_best_psnr += best_seq_psnr / opt.batch_size
        
    return avg_best_mse, avg_best_ssim, avg_best_psnr

def plot_rec(x, epoch, one_hot_action):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]

    if opt.action_repr == 'fc':
        action = action_fc(one_hot_action)
    elif opt.action_repr == 'one_hot':
        action = one_hot_action

    for i in range(1, opt.n_past+opt.n_future):
        if opt.pe_dim != 0:
            h = prior_encoder(x[i-1])
        else:
            h = encoder(x[i-1])
        h_target = encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()

        if opt.action_method == 'skip_posterior':
            z_t = action
        else:
            z_t, _, _ = posterior(h_target)

        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)
   
    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)


# --------- training funtions ------------------------------------
def train(x, one_hot_action):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    if opt.action_repr == 'fc':
        action_fc.zero_grad()

    if opt.pe_dim != 0:
        prior_encoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    if opt.action_repr == 'fc':
        action = action_fc(one_hot_action)
    elif opt.action_repr == 'one_hot':
        action = one_hot_action

    mse = 0
    kld = torch.tensor(0)
    for i in range(1, opt.n_past+opt.n_future):

        if opt.pe_dim != 0:
            h = prior_encoder(x[i-1])
        else:
            h = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h = h[0]

        if opt.action_method == 'skip_posterior':
            z_t = action
        else:
            z_t, mu, logvar = posterior(h_target)

        if opt.action_method == 'concat':
            _, mu_p, logvar_p = prior(torch.cat([h, action], 1))
        elif opt.action_method == 'add':
            # must be FC with matching output size to h.     
            h_and_action = h + action        
            _, mu_p, logvar_p = prior(h_and_action)
        elif opt.action_method == 'no_action':
            _, mu_p, logvar_p = prior(h)
        elif opt.action_method == 'skip_posterior':
            pass
        else:
            raise ValueError('Invalid action method')

        h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])

        if opt.action_method != 'skip_posterior':
            kld += kl_criterion(mu, logvar, mu_p, logvar_p)

    loss = mse
    if opt.action_method != 'skip_posterior':
        loss += kld*opt.beta

    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    if opt.action_repr == 'fc':
        action_fc_optimizer.step()

    if opt.pe_dim != 0:
        prior_encoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

def get_test_metrics(epoch, all_actions = False):
    print("iterating test set of size ", len(test_loader))
    avg_best_mse = np.zeros((opt.n_eval - opt.n_past,))
    avg_best_ssim = np.zeros((opt.n_eval - opt.n_past,))
    avg_best_psnr = np.zeros((opt.n_eval - opt.n_past,))

    for i in range(len(test_loader)):
        test_x, action = next(testing_batch_generator)
        action = action.cuda()

        if i == 0:
            plot_rec(x, epoch, action)

        abm, abss, abp = plot(test_x, action, epoch, metrics_only = (i != 0))

        avg_best_mse += abm/len(test_loader)
        avg_best_ssim += abss/len(test_loader)
        avg_best_psnr += abp/len(test_loader)

        if all_actions and i == 0:
            os.makedirs('%s/gen/%d/' % (opt.log_dir, epoch), exist_ok=True)

            if opt.valid_actions == '':
                action_range = range(opt.action_space)
            else:
                action_range = map(int, opt.valid_actions.split('|'))
                action_range = [x-1 for x in action_range]

            for a in action_range:
                specific_action = torch.zeros_like(action)
                specific_action[:, a] = 1

                plot(test_x, specific_action, epoch, metrics_only = False, not_gt_action = True)

        if i % 10 == 0:
            print("finished test batch", i)
    print("finished iter {}. avg_best_mse: {} avg_best_ssim: {} avg_best_psnr: {}".format(epoch, abm, abss, abp))

    np.savetxt(os.path.join(opt.log_dir, 'mse_ssim_psnr_{}.txt'.format(epoch)), np.vstack((avg_best_mse, \
        avg_best_ssim, avg_best_psnr)), delimiter = ",")

    interval_averages = [epoch, np.average(avg_best_mse), np.average(avg_best_ssim), np.average(avg_best_psnr)]

    f = open(os.path.join(opt.log_dir, 'int_avg_mse_ssim_psnr.txt'), 'a')
    f.write(",".join(map(str, interval_averages)) + "\n")
    f.close()

print('log dir: %s' % opt.log_dir)
print("TOTAL BATCHES", len(train_loader))
# --------- training loop ------------------------------------
for epoch in range(start_epoch, opt.niter):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()

    if opt.action_repr == 'fc':
        action_fc.train()

    if opt.pe_dim != 0:
        prior_encoder.train()

    epoch_mse = 0
    epoch_kld = 0

    epoch_size = len(train_loader)
    progress = progressbar.ProgressBar(max_value=epoch_size).start()

    for i in range(epoch_size):
        progress.update(i+1)

        x, action = next(training_batch_generator)
        action = action.cuda()

        # train frame_predictor 
        mse, kld = train(x, action)
        epoch_mse += mse
        epoch_kld += kld

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/epoch_size, epoch_kld/epoch_size, epoch*epoch_size*opt.batch_size))
    f = open(os.path.join(opt.log_dir, 'train_mse_kld.txt'), 'a')
    write_line = [epoch, epoch_mse/epoch_size, epoch_kld/epoch_size]
    f.write(",".join(map(str, write_line)) + "\n")
    f.close()

    if epoch % 25 == 0:
        # plot some stuff
        frame_predictor.eval()
        #encoder.eval()
        #decoder.eval()
        posterior.eval()
        prior.eval()

        if opt.action_repr == 'fc':
            action_fc.eval()

        all_actions = True
        with torch.no_grad():
            get_test_metrics(epoch, all_actions)

    opt.start_epoch = epoch + 1

    # save the model
    if epoch % 10 == 0 or epoch % 25 == 0:
        save_dict = {
            'encoder': encoder,
            'decoder': decoder,
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'prior': prior,
            'opt': opt}
        
        if opt.action_repr == 'fc':
            save_dict['action_fc'] = action_fc

        if opt.pe_dim != 0:
            save_dict['prior_encoder'] = prior_encoder

        if epoch % 10 == 0:
            torch.save(save_dict,
                '%s/model_latest.pth' % opt.log_dir)
        
        if epoch % 25 == 0:
            torch.save(save_dict,
                '%s/model_%d.pth' % (opt.log_dir, epoch))
