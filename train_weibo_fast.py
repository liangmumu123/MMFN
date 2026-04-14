import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from DWMF import MultiModal
from tqdm import tqdm
import pickle
import os

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

def collate_fn(batch):
    from transformers import BertTokenizer
    import clip
    
    tokenizer = BertTokenizer.from_pretrained('E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/bert-base-chinese')
    
    sents = [item[0] for item in batch]
    image_swin = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # 文本编码
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 32
    epochs = 50
    learning_rate = 3e-5

    print("Loading preprocessed Weibo data...")
    train_dataset = PreprocessedWeiboDataset('E:/MMFN-data/data/weibo/train_preprocessed.pkl')
    test_dataset = PreprocessedWeiboDataset('E:/MMFN-data/data/weibo/test_preprocessed.pkl')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    model = MultiModal(num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    print("\n" + "="*60)
    print("Training on preprocessed Weibo data...")
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
        
        print(f"\nEpoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model_weibo.pth')
            print(f"  >>> Saved best model! Acc: {best_acc:.4f}")

        if test_acc > 0.85:
            print(f"\n🎉 Reached target accuracy! Early stopping...")
            break

    print(f"\n{'='*60}")
    print(f"Training completed! Best accuracy: {best_acc:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
