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
                z_t, _, _ = posterior(h_target)

                if opt.action_method == 'concat':
                    prior(torch.cat([h, action], 1))
                elif opt.action_method == 'add':
                    # must be FC with matching output size to h.                        
                    h_and_action = h + action        
                    prior(h_and_action)
                elif opt.action_method == 'no_action':
                    prior(h)
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