Single GPU
python run_pretrain.py \
        --data_path ../masked/train/ \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 20 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 50 \
        --epochs 1600 \
        --output_dir ../../autodl-tmp
