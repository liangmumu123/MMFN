import torch
import sys
sys.path.append('.')
from MMFN import MultiModal
from gossipcop_dataset import gossipcop_dataset, collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix

device = torch.device('cuda')

# 加载模型
model = MultiModal().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 加载数据
test_set = gossipcop_dataset(is_train=False)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=True)

print('='*60)
print('阈值调优结果')
print('='*60)
print(f'{"阈值":<8} {"realnews召回率":<18} {"fakenews召回率":<18} {"整体准确率":<12}')
print('-'*60)

for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    all_labels = []
    all_preds = []
    
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
            preds = (probs[:, 1] < threshold).long()
            
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    recall_real = recall_score(all_labels, all_preds, pos_label=0)
    recall_fake = recall_score(all_labels, all_preds, pos_label=1)
    acc = accuracy_score(all_labels, all_preds)
    
    print(f'{threshold:<8} {recall_real:<18.4f} {recall_fake:<18.4f} {acc:<12.4f}')

print('='*60)
print('建议: 根据实际需求选择阈值')
print('  - 更关注真实新闻 → 选择较低阈值 (0.35-0.4)')
print('  - 更关注假新闻 → 选择较高阈值 (0.5-0.6)')
