import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import pickle
from transformers import AutoFeatureExtractor
import clip
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

feature_extractor = AutoFeatureExtractor.from_pretrained("E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/swin-base-patch4-window7-224")
clipmodel, preprocess = clip.load('ViT-B/32', device)

for param in clipmodel.parameters():
    param.requires_grad = False

data_root = 'E:/MMFN-data/data/gossipcop'

def preprocess_split(split='train', max_samples=500):
    csv_file = os.path.join(data_root, f'{split}_gossipcop.csv')
    gc = pd.read_csv(csv_file)
    
    all_data = []
    
    print(f"Processing {split} data (first {max_samples} samples)...")
    
    for i in tqdm(range(min(max_samples, len(gc)))):
        img_col = gc.iloc[i]['image'] if 'image' in gc.columns else gc.iloc[i]['images']
        img_name = os.path.basename(img_col)
        label = int(gc.iloc[i]['label'])
        content = str(gc.iloc[i]['content'])
        
        label_folder = 'rumor_images' if label == 0 else 'nonrumor_images'
        img_path = os.path.join(data_root, label_folder, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 提取Swin特征
            swin_input = np.array(img)
            swin_feat = feature_extractor(swin_input, return_tensors="pt").pixel_values.squeeze(0)
            
            # 提取CLIP特征
            clip_input = preprocess(img)
            with torch.no_grad():
                clip_feat = clipmodel.encode_image(clip_input.unsqueeze(0).to(device)).cpu().squeeze(0)
            
            all_data.append({
                'content': content,
                'swin_feat': swin_feat.cpu(),
                'clip_feat': clip_feat,
                'label': label
            })
        except Exception as e:
            print(f"Error: {img_path}")
            continue
    
    # 保存预处理后的数据
    cache_file = os.path.join(data_root, f'{split}_preprocessed.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Saved {len(all_data)} samples to {cache_file}")

if __name__ == '__main__':
    preprocess_split('train', max_samples=500)
    preprocess_split('test', max_samples=200)
    print("预处理完成！")
