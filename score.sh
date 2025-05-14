#!/bin/bash

# 定义所有主体和风格组合
subjects=(
    sbu-dog
    lmei-berry_bowl
    gtou-can
    mmao-cat
    nzhong-clock
    yzi-duck_toy
    gshou-monster_toy
    tyjing-pink_sunglasses
    sshi-poop_emoji
    cche-rc_car
    hemo-red_cartoon
)
styles=(
    "scai-0103:A cat"
    "swei-0301:Slice of watermelon and clouds in the background"
    "scai-0101:A bay"
    "lbi-0105:A village"
    "mtou-0308:A Viking face with beard"
    "fmi-0307:A flower"
    "scai-0104:Flowers"
    "scai-0102:A house"
    "mgu-0206:A mushroom"
    "yhuang-0107:A portrait of a person"
)

# 可选：保存结果到日志文件
LOG_FILE="evaluation_results.log"
echo "Evaluation Results" > "$LOG_FILE"

# 遍历所有组合
for subject in "${subjects[@]}"; do
    subject_id=${subject%-*}
    subject_name=${subject#*-}

    for style in "${styles[@]}"; do
        style_id=${style%%:*}
        style_suffix=${style_id#*-}

        output_folder="./output/${subject_name}-${style_suffix}"

        # 检查输出文件夹是否存在
        if [ ! -d "$output_folder" ]; then
            echo "Output folder $output_folder does not exist. Skipping."
            continue
        fi

        echo "Evaluating: ${subject_name}-${style_suffix}"

        # 内容相似度评估
        content_folder="./dataset-content/${subject_id}"
        echo "Running content evaluation: $content_folder vs $output_folder"
        python score.py --folder1="$content_folder" --folder2="$output_folder"
        content_score=$(python score.py --folder1="$content_folder" --folder2="$output_folder" 2>/dev/null)
        echo "Content CLIP Score: $content_score" >> "$LOG_FILE"

        # 风格相似度评估
        style_folder="./style/${style_suffix}"
        echo "Running style evaluation: $style_folder vs $output_folder"
        python score.py --folder1="$style_folder" --folder2="$output_folder"
        style_score=$(python score.py --folder1="$style_folder" --folder2="$output_folder" 2>/dev/null)
        echo "Style CLIP Score: $style_score" >> "$LOG_FILE"

        style_folder="./style/${style_suffix}"
        echo "Running dino style evaluation: $style_folder vs $output_folder"
        python dino.py --folder_a="$style_folder" --folder_b="$output_folder"
        dino_score=$(python dino.py --folder_a="$style_folder" --folder_b="$output_folder" 2>/dev/null)
        echo "Style DINO Score: $dino_score" >> "$LOG_FILE"

        echo "----------------------------------------" >> "$LOG_FILE"
    done
done

echo "Evaluation complete. Results saved to $LOG_FILE"