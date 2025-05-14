export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export HF_ENDPOINT=https://hf-mirror.com

# 定义主题和风格数组
# subjects=(
#     sbu-dog
#     lmei-berry_bowl
#     gtou-can
#     mmao-cat
#     nzhong-clock
#     yzi-duck_toy
#     gshou-monster_toy
#     tyjing-pink_sunglasses
#     sshi-poop_emoji
#     cche-rc_car
#     hemo-red_cartoon
# )

# 风格数组格式：style_id:subject_description:style_description
# styles=(
#     # "scai-cat_waterpainting-0103:A cat:in watercolor painting style"
#     "swei-3d_rendering-0301:Slice of watermelon and clouds in the background:in 3d rendering style"
#     "scai-bay_waterpainting-0101:A bay:in watercolor painting style"
#     "lbi-crayon-0105:A village:in kid crayon drawing style"
#     "mtou-face_wooden-0308:A Viking face with beard:in wooden sculpture"
#     "fmi-flower_golden3d-0307:A flower:in melting golden 3d rendering style"
#     "scai-flowers_waterpainting-0104:Flowers:in watercolor painting style"
#     "scai-house_waterpainting-0102:A house:in watercolor painting style"
#     "mgu-mushroom_glowing-0206:A mushroom:in glowing style"
#     "yhuang-person_oilpainting-0107:A portrait of a person:in oil painting style"
# )

styles=(
    "scai-0103:A cat"
    # "swei-0301:Slice of watermelon and clouds in the background"、
    # "scai-0101:A bay"
    # "lbi-0105:A village"
    # "mtou-0308:A Viking face with beard"
    # "fmi-0307:A flower"
    "scai-0104:Flowers"
    # "scai-0102:A house"
    # "mgu-0206:A mushroom"
    # "yhuang-0107:A portrait of a person"
)
# # 处理主题训练
# for subject in "${subjects[@]}"; do
#     prefix=$(echo $subject | cut -d'-' -f1)
#     name=$(echo $subject | cut -d'-' -f2-)
    
#     export OUTPUT_DIR="lora-sdxl-$name"
#     export INSTANCE_DIR="./subject/$name"
#     export PROMPT="a $prefix $name"
#     export VALID_PROMPT="a $prefix $name in a bucket"
    
#     echo "开始训练主题: $subject"
#     accelerate launch train_dreambooth_lora_sdxl.py \
#       --pretrained_model_name_or_path=$MODEL_NAME  \
#       --instance_data_dir=$INSTANCE_DIR \
#       --output_dir=$OUTPUT_DIR \
#       --instance_prompt="${PROMPT}" \
#       --rank=8 \
#       --resolution=1024 \
#       --train_batch_size=1 \
#       --learning_rate=5e-5 \
#       --report_to="wandb" \
#       --lr_scheduler="constant" \
#       --lr_warmup_steps=0 \
#       --max_train_steps=1000 \
#       --validation_prompt="${VALID_PROMPT}" \
#       --validation_epochs=50 \
#       --seed="0" \
#       --mixed_precision="no" \
#       --enable_xformers_memory_efficient_attention \
#       --gradient_checkpointing \
#       --use_8bit_adam
#     echo "完成主题: $subject"
# done

# 处理风格训练
for style_entry in "${styles[@]}"; do
    IFS=':' read -r -a parts <<< "$style_entry"
    style_id="${parts[0]}"
    subject_desc="${parts[1]}"
    
    # 从 style_id 提取前缀和名称
    style_prefix=$(echo $style_id | cut -d'-' -f1)
    name=$(echo $style_id | cut -d'-' -f2)  # 修改此行：从 style_id 提取 name
    style_desc="in ${style_prefix} style"
    
    export OUTPUT_DIR="lora-sdxl-$style_id"
    export INSTANCE_DIR="./style/$name"  # 这将指向类似 ./style/3d_rendering
    export PROMPT="${subject_desc} of ${style_desc}"
    export VALID_PROMPT="a person ${style_desc}"
    
    echo "开始训练风格: $style_id"
    accelerate launch train_dreambooth_lora_sdxl.py \
      --pretrained_model_name_or_path=$MODEL_NAME  \
      --instance_data_dir=$INSTANCE_DIR \
      --output_dir=$OUTPUT_DIR \
      --instance_prompt="${PROMPT}" \
      --rank=8 \
      --resolution=1024 \
      --train_batch_size=1 \
      --learning_rate=5e-5 \
      --report_to="wandb" \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=1000 \
      --validation_prompt="${VALID_PROMPT}" \
      --validation_epochs=50 \
      --seed="0" \
      --mixed_precision="no" \
      --enable_xformers_memory_efficient_attention \
      --gradient_checkpointing \
      --use_8bit_adam
    echo "完成风格: $style_id"
done

echo "所有训练任务已完成"