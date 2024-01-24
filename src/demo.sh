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
########## test ###########
# CUDA_VISIBLE_DEVICES=3,2 python main.py --model EAST  --n_GPUs 2 --scale 4 --pre_train '../models/model_best.pt' \
#     --patch_size 64 --save 'efficiency' --n_feats 180 --n_resgroups 7 --n_resblocks 5 --test_only --save_results --dir_data '../../../3D_mix'

# section3.4
# CUDA_VISIBLE_DEVICES=3,2 python main.py --model RCAN  --n_GPUs 2 --dir_data '../../../3Dtestset' --scale 4 \
#     --pre_train '../experiment/RCAN/model/model_best.pt' --save 'testrcan' --data_range '1-300/1-300'\
#     --n_feats 64 --n_resgroups 10 --n_resblocks 20 --patch_size 64 --test_only --save_results
# CUDA_VISIBLE_DEVICES=3,2 python main.py --model SRCNN  --n_GPUs 2 --dir_data '../../../3Dtestset' --scale 4 \
#     --pre_train '../experiment/SRCNN/model/model_best.pt' --save 'testsrcnn' --data_range '1-300/1-300'\
#     --n_feats 64 --n_resgroups 10 --n_resblocks 12 --patch_size 64 --test_only --save_results
# CUDA_VISIBLE_DEVICES=3,2 python main.py --model EDSR  --n_GPUs 2 --dir_data '../../../3Dtestset' --scale 4 \
#     --pre_train '../experiment/EDSR/model/model_best.pt' --save 'testedsr' --data_range '1-300/1-300'\
#     --n_feats 256 --n_resgroups 10 --n_resblocks 32 --patch_size 64 --test_only --save_results
# CUDA_VISIBLE_DEVICES=3,2 python main.py --model EAST  --n_GPUs 2 --dir_data '../../../3Dtestset' --scale 4 \
#     --pre_train '../experiment/EAST/model/model_best.pt' --save 'testEAST' --data_range '1-300/1-300'\
#     --n_feats 180 --n_resgroups 7 --n_resblocks 5 --patch_size 64 --test_only --save_results