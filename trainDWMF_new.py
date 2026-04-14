import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from DWMF import MultiModal
from myweibo_dataset import WeiboDataset, collate_fn
import os

device = torch.device("cpu")
print(f"Using device: {device}")

batch_size = 16
epochs = 10
learning_rate = 2e-5

print("Loading data...")
train_dataset = WeiboDataset('data/weibo/train_weibov.csv')
test_dataset = WeiboDataset('data/weibo/test_weibov.csv')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# 初始化模型
model = MultiModal(num_labels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 断点续训
checkpoint_path = 'checkpoint_epoch2.pth'
start_epoch = 0
best_test_acc = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_test_acc = checkpoint['best_test_acc']
    print(f"✅ 从第 {start_epoch} 轮继续训练，当前最佳准确率: {best_test_acc:.4f}")
else:
    print("从头开始训练")

print("Start training...")
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        image_swin = image_swin.to(device)
        image_clip = image_clip.to(device)
        text_clip = text_clip.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # 测试
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_swin = image_swin.to(device)
            image_clip = image_clip.to(device)
            text_clip = text_clip.to(device)

            logits = model(input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # 保存最佳模型
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  -> 保存最佳模型，准确率: {test_acc:.4f}")

    # 保存断点（每轮结束都保存）
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_acc': best_test_acc
    }, f'checkpoint_epoch{epoch + 1}.pth')
    print(f"  -> 保存断点 checkpoint_epoch{epoch + 1}.pth")

print(f"\n✅ 训练完成！最佳测试准确率: {best_test_acc:.4f}")
print("模型已保存到 best_model.pth")