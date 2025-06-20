OUTPUT_DIR="output/SFs_AU_b24_gpt4o_500k_DisAngle10"
TASK_NAME="ss8"
TARGET_DIR="$OUTPUT_DIR"/downstream_TAPE/$TASK_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3\
    python downstream_TAPE.py \
    --task_name=$TASK_NAME \
    --seed=3 \
    --learning_rate=3e-5 \
    --num_train_epochs=5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=8 \
    --warmup_ratio=0.08 \
    --pretrained_model=ProteinDT \
    --pretrained_folder="$OUTPUT_DIR" --output_dir="$TARGET_DIR" --do_train

#Use OUTPUT_DIR or TARGET_DIR for pretrained_folder
