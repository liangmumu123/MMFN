import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from DWMF import MultiModal
from myweibo_dataset import WeiboDataset, collate_fn
from tqdm import tqdm
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 32
    epochs = 50
    learning_rate = 3e-5

    print("Loading Weibo data...")
    train_dataset = WeiboDataset('E:/MMFN-data/data/weibo/train_weibo_final3.csv')
    test_dataset = WeiboDataset('E:/MMFN-data/data/weibo/test_weibo_final3.csv')

    # 检查标签分布
    train_labels = [train_dataset.df.iloc[i]['label'] for i in range(len(train_dataset))]
    test_labels = [test_dataset.df.iloc[i]['label'] for i in range(len(test_dataset))]
    print(f"训练集 - 标签0: {train_labels.count(0)}, 标签1: {train_labels.count(1)}")
    print(f"测试集 - 标签0: {test_labels.count(0)}, 标签1: {test_labels.count(1)}")

    # 计算类别权重
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / c for c in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"类别权重: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    model = MultiModal(num_labels=2).to(device)
    
    # 使用加权损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 分层学习率
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.swin.parameters(), 'lr': 1e-5},
        {'params': model.image_proj.parameters(), 'lr': 1e-3},
        {'params': model.weight_layer.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], lr=learning_rate, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_acc = 0
    best_results = None

    print("\n" + "="*60)
    print("Training on Weibo dataset with class weights...")
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
            
            # 使用加权损失
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{correct/total:.4f}'})

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        scheduler.step()

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
