import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from DWMF import MultiModal
from myweibo_dataset import WeiboDataset, collate_fn

# 设置设备
device = torch.device("cpu")
print(f"Using device: {device}")

# 训练参数
batch_size = 8
epochs = 2
learning_rate = 2e-5

# 加载数据
print("Loading data...")
full_train_dataset = WeiboDataset('train_weibov.csv')
full_test_dataset = WeiboDataset('test_weibov.csv')

# 只取前500条数据作为训练集，前200条作为测试集
train_size = min(500, len(full_train_dataset))
test_size = min(200, len(full_test_dataset))

train_dataset = Subset(full_train_dataset, list(range(train_size)))
test_dataset = Subset(full_test_dataset, list(range(test_size)))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Train samples: {len(train_dataset)} (fast mode), Test samples: {len(test_dataset)} (fast mode)")

# 初始化模型
model = MultiModal(num_labels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练循环
print("Start training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, token_type_ids, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        # 每20个batch打印一次进度
        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}")
    
    train_acc = correct / total
    train_loss = total_loss / len(train_loader)
    
    # 测试
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_loss_avg = test_loss / len(test_loader)
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss_avg:.4f} | Test Acc: {test_acc:.4f}")

# 最终测试结果
print("\n" + "="*50)
print("FAST TEST RESULTS (500 train, 200 test):")
print(f"Accuracy: {test_acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\n✅ 代码运行成功！可以跑完整数据了。")
