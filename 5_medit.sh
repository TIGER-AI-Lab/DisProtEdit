#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR="output/SFs_AU_b24_gpt4o_500k_DisAngle10"

# Define arrays for tasks and prompt IDs
tasks=("alpha+Villin" "alpha+Pin1" "beta+Villin" "beta+Pin1" "Villin+Pin1")
prompt_ids=(801 802 803 804)

# Loop over each task and prompt ID
for task in "${tasks[@]}"; do
  for prompt_id in "${prompt_ids[@]}"; do
    python3 editing_dis_interpolation.py \
      --editing_task="$task" --text_prompt_id="$prompt_id" --editing_type="both" \
      --decoder_distribution=T5Decoder --score_network_type=T5Base \
      --num_workers=4 --hidden_dim=16 --batch_size=2 \
      --theta=0.9 --num_repeat=16 --oracle_mode=text --AR_generation_mode=01 --AR_condition_mode=expanded \
      --pretrained_folder="$OUTPUT_DIR"
  done
done