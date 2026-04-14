import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import clip
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel, preprocess = clip.load('ViT-B/32', device)

for param in clipmodel.parameters():
    param.requires_grad = False

feature_extractor = AutoFeatureExtractor.from_pretrained("E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/swin-base-patch4-window7-224")
token = BertTokenizer.from_pretrained('E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/bert-base-chinese')


class GossipcopDataset(data.Dataset):
    def __init__(self, is_train=True, data_root='E:/MMFN-data/data/gossipcop'):
        super(GossipcopDataset, self).__init__()
        self.label_dict = []
        self.swin = feature_extractor
        self.preprocess = preprocess
        self.local_path = data_root
        
        if is_train:
            csv_file = os.path.join(self.local_path, 'train_gossipcop.csv')
        else:
            csv_file = os.path.join(self.local_path, 'test_gossipcop.csv')
        
        print(f"Loading data from: {csv_file}")
        gc = pd.read_csv(csv_file)
        
        for i in tqdm(range(len(gc)), desc="Loading data"):
            img_col = gc.iloc[i]['image'] if 'image' in gc.columns else gc.iloc[i]['images']
            img_name = os.path.basename(img_col)
            label = int(gc.iloc[i]['label'])
            content = str(gc.iloc[i]['content'])
            has_image = gc.iloc[i]['has_image'] if 'has_image' in gc.columns else 1
            
            self.label_dict.append({
                'image_name': img_name,
                'label': label,
                'content': content,
                'has_image': has_image
            })
        
        print(f"Loaded {len(self.label_dict)} samples")

    def __getitem__(self, item):
        record = self.label_dict[item]
        img_name = record['image_name']
        label = record['label']
        content = record['content']
        has_image = record['has_image']
        
        label_folder = 'rumor_images' if label == 0 else 'nonrumor_images'
        
        if has_image and img_name:
            full_path = os.path.join(self.local_path, label_folder, img_name)
            try:
                img = Image.open(full_path).convert('RGB')
                img_swin_input = np.array(img)
                img_clip_input = img
            except:
                img_swin_input = np.zeros((224, 224, 3), dtype=np.uint8)
                img_clip_input = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')
        else:
            img_swin_input = np.zeros((224, 224, 3), dtype=np.uint8)
            img_clip_input = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')
        
        # 处理图像
        image_swin = self.swin(img_swin_input, return_tensors="pt").pixel_values.squeeze(0)
        image_clip = self.preprocess(img_clip_input)
        
        return (content, image_swin, image_clip, content), label

    def __len__(self):
        return len(self.label_dict)


def collate_fn(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    imageclip = [i[0][2] for i in data]
    textclip = [i[0][3] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(
        sents,
        truncation=True,
        padding='max_length',
        max_length=300,
        return_tensors='pt'
    )
    
    textclip = clip.tokenize(textclip, truncate=True)
    
    return (data['input_ids'], 
            data['attention_mask'], 
            data['token_type_ids'], 
            torch.stack(image), 
            torch.stack(imageclip), 
            textclip, 
            torch.LongTensor(labels))
