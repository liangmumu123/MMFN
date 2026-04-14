import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import clip
import pickle

warnings.filterwarnings('ignore')

# ============ 模型定义 ============
from transformers import BertModel, SwinModel

class MultiModal(nn.Module):
    def __init__(self, num_labels=2):
        super(MultiModal, self).__init__()
        self.bert = BertModel.from_pretrained('E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/bert-base-chinese')
        self.swin = SwinModel.from_pretrained("E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/swin-base-patch4-window7-224")
        self.image_proj = nn.Linear(1024, 768)
        self.weight_layer = nn.Sequential(
            nn.Linear(768 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, image_swin, image_clip=None, text_clip=None, labels=None):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        text_features = text_outputs.pooler_output
        image_outputs = self.swin(pixel_values=image_swin)
        image_features = image_outputs.pooler_output
        image_proj = self.image_proj(image_features)
        concat_features = torch.cat([text_features, image_proj], dim=1)
        weights = self.weight_layer(concat_features)
        text_weight = weights[:, 0:1]
        image_weight = weights[:, 1:2]
        combined = text_weight * text_features + image_weight * image_proj
        logits = self.classifier(combined)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits

# ============ 加载模型 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

print("加载 best_model_weibo.pth...")
model = MultiModal(num_labels=2)
model.load_state_dict(torch.load("best_model_weibo.pth", map_location=device))
model.to(device)
model.eval()
print("✅ 模型加载成功！")

# ============ 加载 tokenizer 和特征提取器 ============
tokenizer = BertTokenizer.from_pretrained('E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/bert-base-chinese')
feature_extractor = AutoFeatureExtractor.from_pretrained("E:/Dynamic-Weighted-Multi-Modal-Fusion-yuki/swin-base-patch4-window7-224")
clipmodel, preprocess = clip.load('ViT-B/32', device)
clip_tokenize = clip.tokenize

for param in clipmodel.parameters():
    param.requires_grad = False

# ============ 加载测试数据 ============
test_csv = 'E:/MMFN-data/data/weibo/test_weibo_final3.csv'
df = pd.read_csv(test_csv)
print(f"📊 测试集总样本数: {len(df)}")

# 只取前500条快速演示（可根据需要调整）
df = df.head(500)

# ============ 预测函数 ============
def predict_single(text, img_path, label):
    # 文本编码
    encoded = tokenizer(
        text if pd.notna(text) else "",
        truncation=True,
        padding='max_length',
        max_length=300,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device) if 'token_type_ids' in encoded else None
    
    # 图片处理
    try:
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img_name = os.path.basename(img_path)
            alt_path = f"E:/MMFN-data/data/weibo/rumor_images/{img_name}"
            if not os.path.exists(alt_path):
                alt_path = f"E:/MMFN-data/data/weibo/nonrumor_images/{img_name}"
            if os.path.exists(alt_path):
                img = Image.open(alt_path).convert('RGB')
            else:
                img = Image.new('RGB', (224, 224), (255, 255, 255))
        
        # Swin 特征
        img_array = np.array(img)
        swin_input = feature_extractor(img_array, return_tensors="pt").pixel_values.to(device)
        
        # CLIP 特征
        clip_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_clip_features = clipmodel.encode_image(clip_input).float()
            text_clip_features = clipmodel.encode_text(clip_tokenize([text], truncate=True).to(device)).float()
        
    except Exception as e:
        print(f"图片处理错误: {e}")
        swin_input = torch.zeros(1, 3, 224, 224).to(device)
        image_clip_features = torch.zeros(1, 512).to(device)
        text_clip_features = torch.zeros(1, 512).to(device)
    
    # 预测
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids, swin_input, image_clip_features, text_clip_features)
        pred = logits.argmax(1).item()
    
    return pred

# ============ 批量预测 ============
results = []
correct = 0
label_names = {0: "谣言", 1: "非谣言"}

print("\n开始推理...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row['content']) if pd.notna(row['content']) else ""
    img_path = str(row['images'])
    true_label = int(row['label'])
    
    pred = predict_single(text, img_path, true_label)
    
    if pred == true_label:
        correct += 1
    
    results.append({
        '序号': idx + 1,
        '文本': text[:80] + "..." if len(text) > 80 else text,
        '真实标签': label_names[true_label],
        '预测标签': label_names[pred],
        '正确': '✓' if pred == true_label else '✗'
    })

# ============ 输出结果 ============
acc = correct / len(df)
print("\n" + "="*70)
print("🎉 推理完成！")
print("="*70)
print(f"📊 测试样本总数: {len(df)}")
print(f"✅ 正确预测数: {correct}")
print(f"❌ 错误预测数: {len(df) - correct}")
print(f"📈 模型准确率: {acc:.4f} ({acc*100:.2f}%)")
print("="*70)

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 结果已保存到 prediction_results.csv")

# 显示前20条
print("\n📋 前20条预测结果:")
print(results_df.head(20).to_string(index=False))
