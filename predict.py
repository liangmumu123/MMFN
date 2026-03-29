import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import warnings

warnings.filterwarnings('ignore')

# 导入你的模型和数据集
from Core import MultiModal
from myweibo_dataset import weibo_dataset, collate_fn, clipmodel  # 确保从 dataset 导入 clipmodel
from torch.utils.data import DataLoader

device = "cpu"
print(f"✅ 使用设备: {device}")

model = MultiModal()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.to(device)
model.eval()

# 确保 CLIP 也在 cpu 上
clipmodel.to(device)
clipmodel.eval()

print("✅ 模型加载成功！")

test_set = weibo_dataset(is_train=False)
test_loader = DataLoader(
    test_set,
    batch_size=1,
    collate_fn=collate_fn,
    shuffle=False
)

print("\n🔥 开始推理...")
correct = 0
total = 0

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        if idx >= 20: break

        # 1. 解包数据
        # 注意：这里的 clip_img_raw 是预处理后的图片，clip_txt_tokens 是 token 后的序列
        input_ids, att_mask, token_type_ids, imgs, clip_img_raw, clip_txt_tokens, labels = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
        ]

        # 2. 【关键步骤】使用 CLIP 提取特征向量 (512维)
        clip_image_features = clipmodel.encode_image(clip_img_raw).float()  # [1, 512]
        clip_text_features = clipmodel.encode_text(clip_txt_tokens).float()  # [1, 512]

        # 注意参数对应关系：
        # model forward: (input_ids, attention_mask, token_type_ids, image_raw, text, image)
        # 我们传入：      (input_ids, att_mask,     token_type_ids, imgs,      clip_text_features, clip_image_features)
        outputs = model(
            input_ids,
            att_mask,
            token_type_ids,
            imgs,
            clip_text_features,  # 对应模型里的 text 参数
            clip_image_features  # 对应模型里的 image 参数
        )

        predict = torch.argmax(outputs, dim=1)
        total += 1
        if predict.item() == labels.item():
            correct += 1

        print(f"处理第 {idx + 1} 条... 预测: {predict.item()} | 真值: {labels.item()}")

print("\n" + "=" * 50)
print("🎉 推理完成！")
print(f"测试样本：{total}")
print(f"正确：{correct}")
print(f"模型准确率：{correct / total:.4f}")
print("=" * 50)