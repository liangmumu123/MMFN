import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from DWMF import MultiModal
from tqdm import tqdm
import pickle
import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import BertTokenizer, AutoFeatureExtractor

# ============ 1. 预处理函数 ============
def preprocess_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = AutoFeatureExtractor.from_pretrained("E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/swin-base-patch4-window7-224")
    
    data_root = 'E:/MMFN-data/data/weibo'
    
    for split, csv_path in [('train', 'E:/MMFN-data/data/weibo/train_weibo_final3.csv'), 
                            ('test', 'E:/MMFN-data/data/weibo/test_weibo_final3.csv')]:
        
        cache_file = os.path.join(data_root, f'{split}_preprocessed.pkl')
        
        if os.path.exists(cache_file):
            print(f"✅ {split}_preprocessed.pkl 已存在，跳过预处理")
            continue
        
        print(f"\n📦 预处理 {split} 数据...")
        df = pd.read_csv(csv_path)
        
        all_data = []
        
        for idx in tqdm(range(len(df)), desc=split):
            row = df.iloc[idx]
            img_path = str(row['images'])
            label = int(row['label'])
            content = str(row['content']) if 'content' in row and pd.notna(row['content']) else ""
            
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                else:
                    img_name = os.path.basename(img_path)
                    alt_path = f"E:/MMFN-data/data/weibo_images/{img_name}"
                    if os.path.exists(alt_path):
                        img = Image.open(alt_path).convert('RGB')
                    else:
                        img = Image.new('RGB', (224, 224), (255, 255, 255))
                
                img_array = np.array(img)
                swin_feat = feature_extractor(img_array, return_tensors="pt").pixel_values.squeeze(0)
                
            except:
                swin_feat = torch.zeros(3, 224, 224)
            
            all_data.append({
                'content': content,
                'swin_feat': swin_feat.cpu(),
                'label': label
            })
        
        with open(cache_file, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"✅ 保存到 {cache_file}，共 {len(all_data)} 样本")
    
    return True

# ============ 2. 数据集类 ============
class PreprocessedWeiboDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['content'], item['swin_feat'], item['label']

    def __len__(self):
        return len(self.data)

# ============ 3. 训练函数 ============
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")
    
    # 清空显存缓存
    torch.cuda.empty_cache()
    
    tokenizer = BertTokenizer.from_pretrained('E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/bert-base-chinese')
    
    def collate_fn(batch):
        sents = [item[0] for item in batch]
        image_swin = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        encoded = tokenizer(
            sents,
            truncation=True,
            padding='max_length',
            max_length=300,
            return_tensors='pt'
        )
        
        return (encoded['input_ids'],
                encoded['attention_mask'],
                encoded['token_type_ids'] if 'token_type_ids' in encoded else None,
                torch.stack(image_swin),
                torch.LongTensor(labels))
    
    print("\n📊 加载预处理数据...")
    train_dataset = PreprocessedWeiboDataset('E:/MMFN-data/data/weibo/train_preprocessed.pkl')
    test_dataset = PreprocessedWeiboDataset('E:/MMFN-data/data/weibo/test_preprocessed.pkl')
    
    # 显存友好配置
    batch_size = 48  # 从128降到48，适合8GB显存
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Batch size: {batch_size} (每epoch迭代: {len(train_loader)} 次)")
    
    model = MultiModal(num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    epochs = 30
    
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in train_bar:
            input_ids, att_mask, token_type, img_swin, labels = batch
            
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            labels = labels.to(device)
            if token_type is not None:
                token_type = token_type.to(device)
            img_swin = img_swin.to(device)
            
            optimizer.zero_grad()
            loss, logits = model(input_ids, att_mask, token_type, img_swin, labels=labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{correct/total:.4f}'})
        
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        
        # 测试
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids, att_mask, token_type, img_swin, labels = batch
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)
                labels = labels.to(device)
                if token_type is not None:
                    token_type = token_type.to(device)
                img_swin = img_swin.to(device)
                
                logits = model(input_ids, att_mask, token_type, img_swin)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        print(f"\n📊 Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model_weibo.pth')
            print(f"  >>> 💾 Saved best model! Acc: {best_acc:.4f}")
        
        # 每5个epoch清空一次显存
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
        
        if test_acc > 0.85:
            print(f"\n🎉 Reached target accuracy! Early stopping...")
            break
    
    print(f"\n{'='*60}")
    print(f"✅ Training completed! Best accuracy: {best_acc:.4f}")
    print("="*60)
    
    return model

if __name__ == '__main__':
    preprocess_data()
    train()
