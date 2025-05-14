import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
import argparse

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载本地Hugging Face格式的DINO模型
def load_hf_model(model_dir):
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
    model = ViTModel.from_pretrained(model_dir).to(device)
    model.eval()
    return feature_extractor, model

# 初始化模型（请确认路径与文件结构）
feature_extractor, model = load_hf_model("./dino-vits16")

# 自定义预处理流程（与DINO原始论文一致）
def preprocess_images(image_paths):
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)
    return torch.stack(images).to(device)  # 返回batch tensor

def extract_features(model, image_batch):
    with torch.no_grad():
        outputs = model(image_batch)
        features = outputs.last_hidden_state[:, 0, :]  # 取[CLS]标记的特征
        features = F.normalize(features, p=2, dim=1)
    return features

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="计算两个文件夹中图片的平均DINO相似度分数")
    parser.add_argument("--folder_a", type=str, required=True, help="第一个文件夹路径")
    parser.add_argument("--folder_b", type=str, required=True, help="第二个文件夹路径")
    args = parser.parse_args()

    folder_a = args.folder_a
    folder_b = args.folder_b

    # 获取有效图片路径
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    get_image_paths = lambda folder: [
        os.path.join(folder, f) for f in os.listdir(folder) 
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    image_paths_a = get_image_paths(folder_a)
    image_paths_b = get_image_paths(folder_b)
    
    if not image_paths_a or not image_paths_b:
        raise ValueError("两个文件夹都必须包含至少一张有效图片")
    
    # 预处理并提取特征
    batch_a = preprocess_images(image_paths_a)
    batch_b = preprocess_images(image_paths_b)
    
    features_a = extract_features(model, batch_a)
    features_b = extract_features(model, batch_b)
    
    # 计算相似度矩阵并取平均
    similarity_matrix = torch.matmul(features_a, features_b.T)
    average_similarity = similarity_matrix.mean().item()
    
    print(f"DINO score: {average_similarity:.4f}")