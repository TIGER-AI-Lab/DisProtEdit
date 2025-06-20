OUTPUT_DIR="output/SFs_AU_b24_gpt4o_500k_DisAngle10"

CUDA_VISIBLE_DEVICES=0\
        python pretrain_step_04_decoder_SFs.py \
        --batch_size=8 --lr=1e-4 --epochs=10 \
        --decoder_distribution=T5Decoder \
        --score_network_type=T5Base --wandb_name="SFs_AU_b24_gpt4o_500k_DisAngle10"\
        --hidden_dim=16  --verbose \
        --pretrained_folder="$OUTPUT_DIR" \
        --output_model_dir="$OUTPUT_DIR"/step_04_T5 \
        --target_subfolder="pairwise_all"