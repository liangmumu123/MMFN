import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import os

# 加载 BERT tokenizer
token = BertTokenizer.from_pretrained('E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/bert-base-chinese')

# 加载 Swin 特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained("E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/swin-base-patch4-window7-224")

class WeiboDataset(data.Dataset):
    def __init__(self, csv_path, img_root=None):
        super(WeiboDataset, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.swin = feature_extractor
        
        # 如果没指定图片根目录，从路径中提取
        if img_root is None:
            # 从第一张图片路径提取根目录
            first_img = self.df.iloc[0]['images']
            # 提取 D:/PyCharm/projects/my model/data/ 部分
            if '/rumor_images/' in first_img:
                self.img_root = first_img.split('/rumor_images/')[0]
            elif '/nonrumor_images/' in first_img:
                self.img_root = first_img.split('/nonrumor_images/')[0]
            else:
                self.img_root = 'E:/MMFN-data/data/weibo_images'
        else:
            self.img_root = img_root
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Image root: {self.img_root}")
        print(f"Columns: {self.df.columns.tolist()}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 文本内容
        text = str(row['content']) if pd.notna(row['content']) else ""
        
        # 图片路径（已经是完整路径）
        img_path = str(row['images'])
        
        # 标签
        label = int(row['label'])
        
        # 加载图片
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
            else:
                # 尝试使用 img_root 重新构建路径
                img_name = os.path.basename(img_path)
                label_folder = 'rumor_images' if label == 0 else 'nonrumor_images'
                alt_path = os.path.join(self.img_root, label_folder, img_name)
                if os.path.exists(alt_path):
                    img = Image.open(alt_path).convert('RGB')
                else:
                    img = Image.new('RGB', (224, 224), (255, 255, 255))
            
            img_array = np.array(img)
            image_swin = self.swin(img_array, return_tensors="pt").pixel_values.squeeze(0)
        except Exception as e:
            print(f"Error loading image: {img_path}, using zeros")
            image_swin = torch.zeros(3, 224, 224)
        
        # 文本编码
        encoded = token(
            text,
            truncation=True,
            padding='max_length',
            max_length=300,
            return_tensors='pt'
        )
        
        return (
            encoded['input_ids'].squeeze(0),
            encoded['attention_mask'].squeeze(0),
            encoded['token_type_ids'].squeeze(0) if 'token_type_ids' in encoded else None,
            image_swin,
            label
        )

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    
    token_type_ids = None
    if batch[0][2] is not None:
        token_type_ids = torch.stack([item[2] for item in batch])
    
    image_swin = torch.stack([item[3] for item in batch])
    labels = torch.LongTensor([item[4] for item in batch])
    
    return input_ids, attention_mask, token_type_ids, image_swin, labels
