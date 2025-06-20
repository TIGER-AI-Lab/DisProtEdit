OUTPUT_DIR="output/SFs_AU_b24_gpt4o_500k_DisAngle10"

python3 pretrain_step_02_pairwise_representation_SFs.py \
--protein_backbone_model=ProtBERT_BFD \
--batch_size=16 --num_workers=4 \
--pretrained_folder="$OUTPUT_DIR" \
--target_subfolder="pairwise_all" \
--ds_llm="gpt4o"

#--num_samples=5000