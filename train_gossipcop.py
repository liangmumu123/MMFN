import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from DWMF import MultiModal
from gossipcop_dataset import GossipcopDataset, collate_fn, clipmodel
from tqdm import tqdm
import os
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 4
    epochs = 30
    learning_rate = 3e-5
    gradient_accumulation_steps = 16
    data_root = 'E:/MMFN-data/data/gossipcop'

    torch.cuda.empty_cache()
    
    print("Loading data...")
    train_dataset = GossipcopDataset(is_train=True, data_root=data_root)
    test_dataset = GossipcopDataset(is_train=False, data_root=data_root)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = MultiModal(num_labels=2).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

    print("\n" + "="*70)
    print("Start training...")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=100)
        for step, batch in enumerate(train_bar):
            input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_swin = image_swin.to(device)
            image_clip = image_clip.to(device)
            text_clip = text_clip.to(device)

            with torch.no_grad():
                image_clip_encoded = clipmodel.encode_image(image_clip)
                text_clip_encoded = clipmodel.encode_text(text_clip)

            loss, logits = model(input_ids, attention_mask, token_type_ids, image_swin, 
                                image_clip_encoded, text_clip_encoded, labels)
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix({'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 
                                  'Acc': f'{correct/total:.4f}'})

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        scheduler.step()

        # 测试
        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0

        test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test ]", ncols=100)
        with torch.no_grad():
            for batch in test_bar:
                input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip, labels = batch

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                image_swin = image_swin.to(device)
                image_clip = image_clip.to(device)
                text_clip = text_clip.to(device)

                image_clip_encoded = clipmodel.encode_image(image_clip)
                text_clip_encoded = clipmodel.encode_text(text_clip)

                logits = model(input_ids, attention_mask, token_type_ids, image_swin,
                              image_clip_encoded, text_clip_encoded)
                loss = criterion(logits, labels)
                test_loss += loss.item()

                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_loss_avg = test_loss / len(test_loader)
        
        # 计算混淆矩阵和指标
        conf_matrix = confusion_matrix(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        # 每个epoch都输出结果
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs} - Test Results:")
        print(f"Test Accuracy: {test_acc:.6f}")
        print(f"Precision: [{precision[0]:.6f} {precision[1]:.6f}]")
        print(f"Recall: [{recall[0]:.6f} {recall[1]:.6f}]")
        print(f"F1 Score: [{f1[0]:.6f} {f1[1]:.6f}]")
        print(f"\nConfusion Matrix:")
        print(f"[{conf_matrix[0][0]} {conf_matrix[0][1]}]")
        print(f"[{conf_matrix[1][0]} {conf_matrix[1][1]}]")
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['realnews', 'fakenews'], digits=6))
        print(f"{'='*70}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model_gossipcop.pth')
            print(f"\n>>> Saved best model! Accuracy: {best_acc:.6f} <<<\n")

        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Training completed! Best accuracy: {best_acc:.6f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
