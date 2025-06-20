OUTPUT_DIR="output/"

CUDA_VISIBLE_DEVICES=0,1,2,3\
        python3 pretrain_step_01_SFs.py \
        --protein_lr=1e-5 --protein_lr_scale=1 \
        --text_lr=1e-5 --text_lr_scale=1 --CL_loss="EBM_NCE"\
        --protein_backbone_model=ProtBERT_BFD --wandb_name="SFs_AU05_b24_gpt4o_500k"\
        --epochs=10 --batch_size=24 --num_workers=0 --verbose \
        --output_model_dir="$OUTPUT_DIR" --CL=0.0 --D=0.0 --U=0.5 --A=1.0 --dis_angle --ds_llm="gpt4o" --ds_name="500k"
