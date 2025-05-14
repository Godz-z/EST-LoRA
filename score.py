import os
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)

def calculate_embeddings(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image = load_and_preprocess_image(file_path)
            with torch.no_grad():
                image_embedding = model.encode_image(image).squeeze(0)  
            embeddings.append(image_embedding.cpu().numpy())
    return embeddings

def calculate_folder_similarities(folder_pairs):
    results = {}
    for idx, (folder_path1, folder_path2) in enumerate(folder_pairs):
        print(f"\nComparing folder pair {idx+1}: '{folder_path1}' and '{folder_path2}'")
        
        embeddings1 = calculate_embeddings(folder_path1)
        embeddings2 = calculate_embeddings(folder_path2)
        
        if embeddings1 and embeddings2:
            pair_similarity_scores = []
            for embedding1 in embeddings1:
                for embedding2 in embeddings2:
                    score = cosine_similarity([embedding1], [embedding2])[0][0]
                    pair_similarity_scores.append(score)
            
            average_similarity = np.mean(pair_similarity_scores)
            print(f"Average Cosine Similarity for folder pair {idx+1}: {average_similarity:.4f}")
            
            results[(folder_path1, folder_path2)] = {
                "similarity_scores": pair_similarity_scores,
                "average_similarity": average_similarity
            }
        else:
            print(f"Error: One of the folders '{folder_path1}' or '{folder_path2}' contains no valid images.")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate image similarity between two folders using CLIP")
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the second folder")
    args = parser.parse_args()

    folder_pairs = [(args.folder1, args.folder2)]
    results = calculate_folder_similarities(folder_pairs)
    
    if results:
        print("\nFinal Results:")
        for pair, data in results.items():
            print(f"Folder Pair: {pair}")
            print(f"Average Similarity: {data['average_similarity']:.4f}")
            print(f"Individual Scores: {data['similarity_scores']}\n")
            
        overall_avg = np.mean([d['average_similarity'] for d in results.values()])
        print(f"Overall Average Similarity: {overall_avg:.4f}")
    else:
        print("No valid comparisons could be made.")