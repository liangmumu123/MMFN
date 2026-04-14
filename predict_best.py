import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')

from DWMF import MultiModal
from weibo_dataset import WeiboDataset, collate_fn, clipmodel
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    
    # 加载模型
    print("加载 best_model_weibo.pth...")
    model = MultiModal(num_labels=2)
    model.load_state_dict(torch.load("best_model_weibo.pth", map_location=device))
    model.to(device)
    model.eval()
    print("✅ 模型加载成功！")
    
    # 加载测试数据
    test_csv = 'E:/MMFN-data/data/weibo/test_weibo_final3.csv'
    img_root = 'E:/MMFN-data/data/weibo'
    
    print("加载测试数据...")
    test_set = WeiboDataset(root_path=img_root, is_train=False)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn, shuffle=False)
    
    print(f"📊 测试集总样本数: {len(test_set)}")
    print("\n" + "="*80)
    print("开始推理...")
    print("="*80)
    
    results = []
    correct = 0
    total = 0
    label_names = {0: "谣言", 1: "非谣言"}
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="推理进度")):
            input_ids, att_mask, token_type, swin_img, clip_img, clip_text, labels = batch
            
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            labels = labels.to(device)
            if token_type is not None:
                token_type = token_type.to(device)
            swin_img = swin_img.to(device)
            clip_img = clip_img.to(device)
            clip_text = clip_text.to(device)
            
            # 提取 CLIP 特征
            image_clip_features = clipmodel.encode_image(clip_img).float()
            text_clip_features = clipmodel.encode_text(clip_text).float()
            
            # 预测
            logits = model(input_ids, att_mask, token_type, swin_img,
                          image_clip_features, text_clip_features)
            pred = logits.argmax(1).item()
            
            true_label = labels.item()
            if pred == true_label:
                correct += 1
            total += 1
            
            # 获取文本
            text_content = test_set.label_dict[idx]['content']
            if len(text_content) > 100:
                text_content = text_content[:100] + "..."
            
            results.append({
                '序号': idx + 1,
                '文本内容': text_content,
                '真实标签': label_names[true_label],
                '预测标签': label_names[pred],
                '是否正确': '✓' if pred == true_label else '✗'
            })
            
            # 每100条打印一次
            if (idx + 1) % 100 == 0:
                current_acc = correct / total
                print(f"已处理 {idx+1}/{len(test_set)} 条, 当前准确率: {current_acc:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
    
    # 最终统计
    final_acc = correct / total
    print("\n" + "="*80)
    print("🎉 推理完成！")
    print("="*80)
    print(f"📊 测试样本总数: {total}")
    print(f"✅ 正确预测数: {correct}")
    print(f"❌ 错误预测数: {total - correct}")
    print(f"📈 模型准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print("="*80)
    
    # 显示前20条结果
    print("\n📋 前20条预测结果:")
    print("-"*80)
    print(results_df.head(20).to_string(index=False))
    print("="*80)
    
    print(f"\n✅ 完整结果已保存到 prediction_results.csv")

if __name__ == '__main__':
    main()
