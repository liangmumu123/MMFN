import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from DWMF import MultiModal
from myweibo_dataset import WeiboDataset, collate_fn
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 16
    epochs = 50
    learning_rate = 2e-5
    weight_decay = 0.01
    gradient_accumulation_steps = 4

    scaler = GradScaler()
    torch.cuda.empty_cache()
    
    print("Loading Weibo data from E:/MMFN-data/data/weibo...")
    
    # 使用正确的 weibo 数据路径
    train_dataset = WeiboDataset('E:/MMFN-data/data/weibo/train_weibo_final3.csv')
    test_dataset = WeiboDataset('E:/MMFN-data/data/weibo/test_weibo_final3.csv')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = MultiModal(num_labels=2).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.swin.parameters(), 'lr': 1e-5},
        {'params': model.image_proj.parameters(), 'lr': 1e-3},
        {'params': model.weight_layer.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    
    def smooth_loss(logits, labels, smoothing=0.1):
        n_classes = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_classes - 1)
        log_probs = nn.LogSoftmax(dim=1)(logits)
        loss = -(smooth_one_hot * log_probs).sum(dim=1).mean()
        return loss

    best_acc = 0
    best_results = None
    patience = 10
    patience_counter = 0

    print("\n" + "="*70)
    print("Start training on Weibo dataset...")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=100)
        for step, batch in enumerate(train_bar):
            input_ids, attention_mask, token_type_ids, image_swin, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_swin = image_swin.to(device)

            with autocast():
                loss, logits = model(input_ids, attention_mask, token_type_ids, image_swin, 
                                    labels=labels)
                loss = smooth_loss(logits, labels, smoothing=0.1)
            
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix({'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 
                                  'Acc': f'{correct/total:.4f}'})

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0

        test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test ]", ncols=100)
        with torch.no_grad():
            for batch in test_bar:
                input_ids, attention_mask, token_type_ids, image_swin, labels = batch

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                image_swin = image_swin.to(device)

                logits = model(input_ids, attention_mask, token_type_ids, image_swin)
                loss = criterion(logits, labels)
                test_loss += loss.item()

                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_loss_avg = test_loss / len(test_loader)
        
        precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs} - Results:")
        print(f"Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f}")
        print(f"Test Acc: {test_acc:.4f} | Test Loss: {test_loss_avg:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Precision: [{precision[0]:.4f} {precision[1]:.4f}]")
        print(f"Recall: [{recall[0]:.4f} {recall[1]:.4f}]")
        print(f"F1 Score: [{f1[0]:.4f} {f1[1]:.4f}]")
        print(f"\nConfusion Matrix:")
        print(f"[{conf_matrix[0][0]} {conf_matrix[0][1]}]")
        print(f"[{conf_matrix[1][0]} {conf_matrix[1][1]}]")
        print(f"{'='*70}")

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            best_results = {
                'accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix,
                'all_preds': all_preds,
                'all_labels': all_labels,
                'epoch': epoch + 1
            }
            torch.save(model.state_dict(), 'best_model_weibo.pth')
            print(f"\n>>> ★ Saved best model! Accuracy: {best_acc:.4f} ★ <<<\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()

    if best_results:
        print("\n" + "="*70)
        print("FINAL BEST RESULTS ON WEIBO:")
        print(f"Best Epoch: {best_results['epoch']}")
        print(f"Overall Accuracy: {best_results['accuracy']:.4f}")
        print(f"Precision per class: {best_results['precision'].tolist()}")
        print(f"Recall per class: {best_results['recall'].tolist()}")
        print(f"F1 Score per class: {best_results['f1'].tolist()}")
        print("Confusion Matrix:")
        print(best_results['confusion_matrix'])
        print("\nClassification Report:")
        print(classification_report(best_results['all_labels'], best_results['all_preds'], 
                                    target_names=['realnews', 'fakenews'], digits=4))
        print("="*70)

if __name__ == '__main__':
    main()
