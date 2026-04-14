import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from DWMF import MultiModal
from myweibo_dataset import WeiboDataset, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# 使用 gossipcop 训练的模型
model_file = "best_model_gossipcop.pth"
print(f"📂 加载模型: {model_file}")

# 加载模型
model = MultiModal(num_labels=2)
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

print("✅ 模型加载成功！")

# 加载测试数据
test_set = WeiboDataset('E:/MMFN-data/data/weibo/test_weibo_final3.csv')
test_loader = DataLoader(
    test_set,
    batch_size=1,
    collate_fn=collate_fn,
    shuffle=False
)

print(f"📊 测试集总样本数: {len(test_set)}")
print("\n" + "="*80)
print("开始推理...")
print("="*80)

results = []
correct = 0
total = 0

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        input_ids, att_mask, token_type_ids, image_swin, labels = batch
        
        input_ids = input_ids.to(device)
        att_mask = att_mask.to(device)
        labels = labels.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        image_swin = image_swin.to(device)
        
        outputs = model(input_ids, att_mask, token_type_ids, image_swin)
        predict = torch.argmax(outputs, dim=1)
        
        pred_label = predict.item()
        true_label = labels.item()
        
        label_names = {0: "谣言", 1: "非谣言"}
        
        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1
        total += 1
        
        # 修复：将文本内容转换为字符串，处理 NaN 情况
        text_content_raw = test_set.df.iloc[idx]['content']
        if pd.isna(text_content_raw):
            text_content = ""
        else:
            text_content = str(text_content_raw)
        
        # 存储结果
        result = {
            '序号': idx + 1,
            '文本内容': text_content[:100] + "..." if len(text_content) > 100 else text_content,
            '真实标签': label_names[true_label],
            '预测标签': label_names[pred_label],
            '是否正确': "✓ 正确" if is_correct else "✗ 错误",
            '真实标签值': true_label,
            '预测标签值': pred_label
        }
        results.append(result)
        
        # 打印详细结果
        print(f"\n{'='*80}")
        print(f"📝 样本 {idx + 1}/{len(test_set)}")
        if len(text_content) > 150:
            print(f"📄 文本内容: {text_content[:150]}...")
        elif text_content:
            print(f"📄 文本内容: {text_content}")
        else:
            print(f"📄 文本内容: [无文本内容]")
        print(f"🏷️  真实标签: {label_names[true_label]} ({true_label})")
        print(f"🤖 预测标签: {label_names[pred_label]} ({pred_label})")
        print(f"{'✅ 正确' if is_correct else '❌ 错误'}")
        
        # 每预测50个样本显示进度
        if (idx + 1) % 50 == 0:
            current_acc = correct / total
            print(f"\n📈 当前进度: {idx + 1}/{len(test_set)}, 当前准确率: {current_acc:.4f}")

# 保存结果到 CSV
results_df = pd.DataFrame(results)
results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 结果已保存到 prediction_results.csv")

# 最终统计
print("\n" + "="*80)
print("🎉 推理完成！")
print("="*80)
print(f"📊 测试样本总数: {total}")
print(f"✅ 正确预测数: {correct}")
print(f"❌ 错误预测数: {total - correct}")
print(f"📈 模型准确率: {correct / total:.4f} ({correct/total*100:.2f}%)")
print("="*80)

# 显示错误预测的样本
error_results = [r for r in results if r['是否正确'] == "✗ 错误"]
if error_results:
    print(f"\n⚠️ 错误预测样本 (共 {len(error_results)} 个，显示前10个):")
    print("-"*80)
    for err in error_results[:10]:
        print(f"序号 {err['序号']}: 真实={err['真实标签']}, 预测={err['预测标签']}")
        print(f"  文本: {err['文本内容'][:80]}...")
        print()
else:
    print("\n🎉 完美！没有错误预测！")

print("="*80)
