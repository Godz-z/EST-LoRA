#!/bin/bash

# 定义所有主体和风格组合
subjects=(
    sbu-dog
    lmei-berry_bowl
    # gtou-can
    # mmao-cat
    nzhong-clock
    # yzi-duck_toy
    # gshou-monster_toy
    # tyjing-pink_sunglasses
    # sshi-poop_emoji
    # cche-rc_car
    # hemo-red_cartoon
)

styles=(
    "scai-0103:A cat"
    # "swei-0301:Slice of watermelon and clouds in the background"
    # "scai-0101:A bay"
    # "lbi-0105:A village"
    # "mtou-0308:A Viking face with beard"
    # "fmi-0307:A flower"
    "scai-0104:Flowers"
    # "scai-0102:A house"
    # "mgu-0206:A mushroom"
    # "yhuang-0107:A portrait of a person"
)

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export HF_ENDPOINT=https://hf-mirror.com

# 遍历所有主体
for subject in "${subjects[@]}"; do
    # 解析主体信息
    subject_id=${subject%-*}
    subject_name=${subject#*-}
    display_name=$(echo "$subject" | tr '-' ' ')
    
    # 遍历所有风格
    for style in "${styles[@]}"; do
        # 解析风格信息
        style_id=${style%%:*}
        style_desc=${style#*:}
        style_prefix=${style_id%-*}
        style_suffix=${style_id#*-}
        
        # 设置路径参数
        export LORA_PATH_CONTENT="./content/lora-sdxl-${subject_name}"
        export LORA_PATH_STYLE="./lora-sdxl-${style_id}"
        export OUTPUT_FOLDER="./output1/${subject_name}-${style_suffix}"
        export PROMPT="a ${display_name} in ${style_prefix} style"
        
        # 创建输出目录
        mkdir -p "$OUTPUT_FOLDER"
        
        # 执行生成命令
        python inference_sd.py \
          --pretrained_model_name_or_path="$MODEL_NAME" \
          --lora_name_or_path_content="$LORA_PATH_CONTENT" \
          --lora_name_or_path_style="$LORA_PATH_STYLE" \
          --output_folder="$OUTPUT_FOLDER" \
          --prompt="$PROMPT"
    done
done