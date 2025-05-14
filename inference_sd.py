import argparse
from diffusers import DiffusionPipeline
from transformers import ViTFeatureExtractor, ViTModel
import torch
import os
from utils import insert_sd_klora_to_unet
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--lora_name_or_path_content",
        type=str,
        help="LoRA path",
        default="loraDataset/content_6/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--lora_name_or_path_style",
        type=str,
        help="LoRA path",
        default="loraDataset/style_9/pytorch_lora_weights.safetensors",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the image generation",
        default="a sbu cat in szn style",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern for the image generation",
        default="s",
    )
    return parser.parse_args()


args = parse_args()
pattern = args.pattern
if pattern == "s*":
    alpha = 1.5
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = alpha * 0.5
    
sum_timesteps = 28000


device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cuda" if torch.cuda.is_available() else "cpu"
def test():
    """生成测试图片并计算CLIP相似度（自动释放资源）"""
    test_seed = 0  # 固定种子
    
    # 创建content测试管道
    pipe_content = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipe_content.load_lora_weights(args.lora_name_or_path_content)
    pipe_content.to(device, torch.float16)
    
    generator = torch.Generator(device=device).manual_seed(test_seed)
    image_content = pipe_content(
        prompt=args.prompt,
        generator=generator
    ).images[0]
    
    output_path_content = os.path.join(args.output_folder, "test-content.png")
    image_content.save(output_path_content)
    print(f"Saved: {output_path_content}")
    
    # 创建style测试管道
    pipe_style = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipe_style.load_lora_weights(args.lora_name_or_path_style)
    pipe_style.to(device, torch.float16)
    
    generator = torch.Generator(device=device).manual_seed(test_seed)
    image_style = pipe_style(
        prompt=args.prompt,
        generator=generator
    ).images[0]
    
    output_path_style = os.path.join(args.output_folder, "test-style.png")
    image_style.save(output_path_style)
    print(f"Saved: {output_path_style}")
    
    # 释放生成模型的显存
    del pipe_content, pipe_style
    torch.cuda.empty_cache()

    # ==== 新增DINO相似度计算部分（自动释放资源）====
    print("\nCalculating DINO similarity between generated images...")
    
    try:
        # 加载DINO模型和特征提取器
        feature_extractor = ViTFeatureExtractor.from_pretrained("./dino-vits16")
        model = ViTModel.from_pretrained("./dino-vits16").to(device)
        
        def load_image(path):
            image = Image.open(path).convert("RGB")
            inputs = feature_extractor(images=image, return_tensors="pt").to(device)
            return inputs
        
        content_inputs = load_image(output_path_content)
        style_inputs = load_image(output_path_style)
        
        with torch.no_grad():
            content_outputs = model(**content_inputs)
            style_outputs = model(**style_inputs)
            
            # 提取最后一层的 [CLS] token 特征作为图像表示
            content_embedding = content_outputs.last_hidden_state[:, 0, :].squeeze()
            style_embedding = style_outputs.last_hidden_state[:, 0, :].squeeze()
        
        # 计算余弦相似度
        cosine_sim = cosine_similarity(
            content_embedding.cpu().numpy().reshape(1, -1),
            style_embedding.cpu().numpy().reshape(1, -1)
        )[0][0]
        print(f"DINO Similarity Score: {cosine_sim:.4f}")
        
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
    finally:
        # 显式释放DINO模型和临时变量
        del model, content_inputs, style_inputs
        torch.cuda.empty_cache()  # 释放显存
        
    print("Resources released successfully")
    return cosine_sim
    print("Resources released successfully")
    return cosine_sim 
def run(score):
# def run():
    """主运行函数"""
    # 2. 测试完成后加载主程序的管道（此时测试管道已释放）
    score = 1-score
    print('beta', score)
    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipe.unet = insert_sd_klora_to_unet(
        pipe.unet, 
        args.lora_name_or_path_content, 
        args.lora_name_or_path_style, 
        alpha, score, sum_timesteps, pattern
    )
    pipe.to(device, dtype=torch.float16)
    
    # 3. 执行批量生成
    seeds = list(range(40))
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=args.prompt,
            generator=generator
        ).images[0]
        
        output_path = os.path.join(
            args.output_folder, 
            f"output_image_{index}.png"
        )
        image.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    score = test()
    run(score)
    # run()
