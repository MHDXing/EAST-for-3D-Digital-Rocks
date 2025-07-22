
# 20240918
# train
# CUDA_VISIBLE_DEVICES=3,2,1,0  python main.py --dir_data '../../../3D_mix' --model EAST --save '0918train' --n_GPUs 2 --scale 4 \
#     --save_results --epochs 1 --batch_size 16 --patch_size 64 --warm_up 10 --gclip 20 --reset --window_sizes '2-4-8'\
#     --lr 1e-4 --test_every 12 --n_feats 180 --n_resgroups 7 --n_resblocks 5 --print_every 50 \
#     --loss '1*Charbonnier+2*HF' --optimizer 'AdamW' --noise

# test
CUDA_VISIBLE_DEVICES=3,2 python main.py --model EAST  --n_GPUs 1 --dir_data '../../../3Dtestset' --scale 4 \
    --pre_train '../models/model_best.pt' --save '0918test' --data_range '1-300/1-300'\
    --n_feats 180 --n_resgroups 7 --n_resblocks 5 --patch_size 64 --test_only --save_results