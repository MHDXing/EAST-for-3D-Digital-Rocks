CUDA_VISIBLE_DEVICES=3,2,1,0  python main.py --dir_data '../../../3D_mix' --model EAST --save EAST_ws448 --n_GPUs 2 --scale 4 \
    --save_results --epochs 300 --decay '200-300-400-500' --batch_size 16 --patch_size 64 --warm_up 10 --gclip 20 --reset --window_sizes '4-4-8'\
    --lr 2e-4 --test_every 6 --n_feats 180 --n_resgroups 7 --n_resblocks 5 --print_every 50 --loss '1*Charbonnier+2*HF' --noise --optimizer 'AdamW' #--no_augment
# CUDA_VISIBLE_DEVICES=3,2,1,0  python main.py --dir_data '../../../3D_mix' --model EAST --save EAST_ws488 --n_GPUs 2 --scale 4 \
#     --save_results --epochs 300 --decay '200-300-400-500' --batch_size 16 --patch_size 64 --warm_up 10 --gclip 20 --reset --window_sizes '4-8-8'\
#     --lr 2e-4 --test_every 6 --n_feats 180 --n_resgroups 7 --n_resblocks 5 --print_every 50 --loss '1*Charbonnier+2*HF' --noise #--no_augment
# CUDA_VISIBLE_DEVICES=3,2,1,0  python main.py --dir_data '../../../3D_mix' --model SRCNN --save SRCNN --n_GPUs 1 --scale 4 \
#     --save_results --epochs 600 --decay '100-200-300-400-500' --batch_size 16 --patch_size 64 --warm_up 0 --gclip 10 --reset \
#     --lr 1e-4 --test_every 12 --n_feats 64 --n_resblocks 12 --print_every 50 --loss '1*L1' --noise #--no_augment
# CUDA_VISIBLE_DEVICES=3,2,1,0  python main.py --dir_data '../../../3D_mix' --model RCAN --save RCAN --n_GPUs 4 --scale 4 \
#     --save_results --epochs 600 --decay '100-200-300-400-500' --batch_size 16 --patch_size 64 --warm_up 0 --gclip 10 --reset \
#     --lr 1e-4 --test_every 6 --n_feats 64 --n_resgroups 10 --n_resblocks 20 --print_every 50 --loss '1*L1' --noise


# train
# CUDA_VISIBLE_DEVICES=1,0  python main.py --dir_data '../../../3D_mix' --model EAST --save EAST --n_GPUs 2 --scale 4 \
#     --save_results --epochs 600 --batch_size 16 --patch_size 64 --warm_up 10 --gclip 20 --reset --window_sizes '4-8-8'\
#     --lr 1e-4 --test_every 12 --n_feats 180 --n_resgroups 7 --n_resblocks 5 --print_every 50 \
#     --loss '1*Charbonnier+2*HF' --optimizer 'AdamW' --noise

# test
CUDA_VISIBLE_DEVICES=0,1 python main.py --model EAST  --n_GPUs 1 --dir_data '../../../3Dtestset' --scale 4 \
    --pre_train '../models/model_best.pt' --save 'result' --data_range '1-300/1-300'\
    --n_feats 180 --n_resgroups 7 --n_resblocks 5 --patch_size 64 --test_only --save_results
