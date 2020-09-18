python train_svg_lp.py --dataset 50salads --model vgg --g_dim 128 \
--z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 \
--data_root /scratch/ssd001/home/winstonc/datasets/50salads-n_pairs=None-mse=30.00-action_pair=None/ \
--log_dir /scratch/hdd001/home/winstonc/logs/svg/ --annotations_file grabbing_actions_only_20 \
--n_eval 20 --batch_size 10 --max_dataset_size 100 --epoch_size 10