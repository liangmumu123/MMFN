import torch
from torch.utils.data import DataLoader
from DWMF import MultiModal
from myweibo_dataset import WeiboDataset, collate_fn

device = torch.device("cpu")
print(f"Using device: {device}")

# 加载测试数据
print("Loading test data...")
test_dataset = WeiboDataset('data/weibo/test_weibov.csv')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
print(f"Test samples: {len(test_dataset)}")

# 加载模型
print("Loading model...")
model = MultiModal(num_labels=2).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("Model loaded successfully!")

# 推理
print("\n开始推理...")
correct = 0
total = 0

for i, batch in enumerate(test_loader):
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
        logits = model(input_ids, attention_mask, token_type_ids, image_swin, image_clip, text_clip)
        pred = logits.argmax(dim=1).item()

    true_label = labels.item()
    print(f"处理第 {i + 1} 条 | 真值：{true_label} | 预测：{pred}")

    if pred == true_label:
        correct += 1
    total += 1

accuracy = correct / total
print(f"\n推理完成！")
print(f"测试样本：{total}")
print(f"正确：{correct}")
print(f"模型准确率：{accuracy:.4f}")