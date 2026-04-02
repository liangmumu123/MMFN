import torch
import sys
sys.path.append('.')

from MMFN import MultiModal
from gossipcop_dataset import gossipcop_dataset, collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("="*60)
print("测试最佳模型")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 加载模型
print("\n1. 加载最佳模型...")
model = MultiModal().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
print("   ✅ 模型加载成功")

# 2. 加载测试数据
print("\n2. 加载测试数据...")
test_set = gossipcop_dataset(is_train=False)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=True)
print(f"   ✅ 测试集大小: {len(test_set)}")
print(f"   ✅ 实际测试 batch 数: {len(test_loader)}")

# 3. 测试
print("\n3. 开始测试...")
all_labels = []
all_preds = []

with torch.no_grad():
    for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(test_loader):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        image = image.cuda()
        imageclip = imageclip.cuda()
        textclip = textclip.cuda()
        
        output = model(input_ids, attention_mask, token_type_ids, image, textclip, imageclip)
        preds = output.argmax(1)
        
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        if (i + 1) % 100 == 0:
            print(f"   已测试 {i+1} 个 batch")

# 4. 计算指标
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average=None)
rec = recall_score(all_labels, all_preds, average=None)
f1 = f1_score(all_labels, all_preds, average=None)
conf = confusion_matrix(all_labels, all_preds)

print("\n" + "="*60)
print("测试结果")
print("="*60)
print(f"\n整体准确率: {acc:.4f} ({acc*100:.2f}%)")
print(f"\n各类别指标:")
print(f"  realnews  - 精确率: {prec[0]:.4f}, 召回率: {rec[0]:.4f}, F1: {f1[0]:.4f}")
print(f"  fakenews  - 精确率: {prec[1]:.4f}, 召回率: {rec[1]:.4f}, F1: {f1[1]:.4f}")
print(f"\n混淆矩阵:")
print(f"              预测 realnews  预测 fakenews")
print(f"实际 realnews      {conf[0][0]:>4}           {conf[0][1]:>4}")
print(f"实际 fakenews      {conf[1][0]:>4}           {conf[1][1]:>4}")

realnews_correct = conf[0][0]
fakenews_correct = conf[1][1]
realnews_total = conf[0][0] + conf[0][1]
fakenews_total = conf[1][0] + conf[1][1]

print(f"\n详细统计:")
print(f"  realnews: {realnews_correct}/{realnews_total} 正确 ({realnews_correct/realnews_total*100:.1f}%)")
print(f"  fakenews: {fakenews_correct}/{fakenews_total} 正确 ({fakenews_correct/fakenews_total*100:.1f}%)")

print("\n" + "="*60)
print("✅ 测试完成")
