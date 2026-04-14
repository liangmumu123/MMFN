import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import pickle
import pandas as pd
from transformers import AutoFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

feature_extractor = AutoFeatureExtractor.from_pretrained("E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/swin-base-patch4-window7-224")

data_root = 'E:/MMFN-data/data/weibo'

def preprocess_weibo_data(csv_path, split_name, max_samples=None):
    """预处理 weibo 数据，提取并保存 Swin 特征"""
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df[:max_samples]
    
    all_data = []
    
    print(f"Processing {split_name} data ({len(df)} samples)...")
    
    for idx in tqdm(range(len(df)), desc=split_name):
        row = df.iloc[idx]
        
        # 获取图片路径
        img_path = str(row['images'])
        label = int(row['label'])
        
        # 获取文本
        content = str(row['content']) if 'content' in row and pd.notna(row['content']) else ""
        
        # 加载并处理图片
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
            else:
                # 尝试替代路径
                img_name = os.path.basename(img_path)
                alt_path = f"E:/MMFN-data/data/weibo_images/{img_name}"
                if os.path.exists(alt_path):
                    img = Image.open(alt_path).convert('RGB')
                else:
                    img = Image.new('RGB', (224, 224), (255, 255, 255))
            
            img_array = np.array(img)
            swin_feat = feature_extractor(img_array, return_tensors="pt").pixel_values.squeeze(0)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            swin_feat = torch.zeros(3, 224, 224)
        
        all_data.append({
            'content': content,
            'swin_feat': swin_feat.cpu(),
            'label': label
        })
    
    # 保存预处理数据
    cache_file = os.path.join(data_root, f'{split_name}_preprocessed.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Saved {len(all_data)} samples to {cache_file}")
    
    return all_data

if __name__ == '__main__':
    # 先测试小数据集
    preprocess_weibo_data('E:/MMFN-data/data/weibo/train_weibo_final3.csv', 'train_small', max_samples=500)
    preprocess_weibo_data('E:/MMFN-data/data/weibo/test_weibo_final3.csv', 'test_small', max_samples=200)
    print("\n预处理完成！")
