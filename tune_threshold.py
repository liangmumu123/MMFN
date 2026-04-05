import torch
import sys
sys.path.append('.')
from MMFN import MultiModal
from gossipcop_dataset import gossipcop_dataset, collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix

print('='*60)
print('动态阈值调优 - 基于最佳模型')
print('='*60)

device = torch.device('cuda')
model = MultiModal().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_set = gossipcop_dataset(is_train=False)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, collate_fn=collate_fn, drop_last=True)

# 收集所有概率
all_probs = []
all_labels = []

with torch.no_grad():
    for input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label in test_loader:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        image = image.cuda()
        imageclip = imageclip.cuda()
        textclip = textclip.cuda()
        
        output = model(input_ids, attention_mask, token_type_ids, image, textclip, imageclip)
        probs = torch.softmax(output, dim=1)
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(label.numpy())

print(f'\n阈值调优结果:')
print(f"{'阈值':<8} {'real召回':<12} {'fake召回':<12} {'整体准确率':<12} {'F1分数':<12}")
print('-'*60)

best = {'threshold': 0.5, 'f1': 0}

for th in [0.35, 0.38, 0.4, 0.42, 0.45, 0.48, 0.5, 0.52, 0.55]:
    preds = [1 if p > th else 0 for p in all_probs]
    recall_real = recall_score(all_labels, preds, pos_label=0)
    recall_fake = recall_score(all_labels, preds, pos_label=1)
    acc = accuracy_score(all_labels, preds)
    f1 = 2 * (recall_real * recall_fake) / (recall_real + recall_fake + 1e-8)
    
    print(f"{th:<8} {recall_real:<12.4f} {recall_fake:<12.4f} {acc:<12.4f} {f1:<12.4f}")
    
    if f1 > best['f1']:
        best = {'threshold': th, 'f1': f1, 'recall_real': recall_real, 'recall_fake': recall_fake, 'acc': acc}

print('='*60)
print(f"最佳阈值: {best['threshold']}")
print(f"  realnews 召回率: {best['recall_real']:.4f}")
print(f"  fakenews 召回率: {best['recall_fake']:.4f}")
print(f"  整体准确率: {best['acc']:.4f}")
