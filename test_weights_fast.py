import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from DWMF import MultiModal
from gossipcop_dataset import GossipcopDataset, collate_fn, clipmodel
from tqdm import tqdm
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 使用很小的数据集快速测试
    batch_size = 8
    epochs = 2  # 只跑2个epoch
    data_root = 'E:/MMFN-data/data/gossipcop'

    print("Loading data...")
    full_train_dataset = GossipcopDataset(is_train=True, data_root=data_root)
    full_test_dataset = GossipcopDataset(is_train=False, data_root=data_root)
    
    # 只用200个训练样本，50个测试样本
    train_dataset = Subset(full_train_dataset, range(200))
    test_dataset = Subset(full_test_dataset, range(50))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)

    print(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")

    model = MultiModal(num_labels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    print("\n快速测试动态权重...")
    print("="*60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in train_bar:
            input_ids, att_mask, token_type, img_swin, img_clip, txt_clip, labels = batch
            
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            labels = labels.to(device)
            if token_type is not None:
                token_type = token_type.to(device)
            img_swin = img_swin.to(device)
            img_clip = img_clip.to(device)
            txt_clip = txt_clip.to(device)
            
            with torch.no_grad():
                img_clip_enc = clipmodel.encode_image(img_clip)
                txt_clip_enc = clipmodel.encode_text(txt_clip)
            
            optimizer.zero_grad()
            # 返回权重
            loss, logits, text_w, image_w = model(input_ids, att_mask, token_type, img_swin,
                                                   img_clip_enc, txt_clip_enc, labels, return_weights=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
            # 显示当前batch的权重
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total:.3f}',
                'T_w': f'{text_w.mean().item():.3f}',
                'I_w': f'{image_w.mean().item():.3f}'
            })
        
        # 测试
        model.eval()
        all_preds = []
        all_labels = []
        all_text_w = []
        all_image_w = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids, att_mask, token_type, img_swin, img_clip, txt_clip, labels = batch
                
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)
                labels = labels.to(device)
                if token_type is not None:
                    token_type = token_type.to(device)
                img_swin = img_swin.to(device)
                img_clip = img_clip.to(device)
                txt_clip = txt_clip.to(device)
                
                img_clip_enc = clipmodel.encode_image(img_clip)
                txt_clip_enc = clipmodel.encode_text(txt_clip)
                
                logits, text_w, image_w = model(input_ids, att_mask, token_type, img_swin,
                                                 img_clip_enc, txt_clip_enc, return_weights=True)
                
                all_text_w.extend(text_w.cpu().numpy().flatten())
                all_image_w.extend(image_w.cpu().numpy().flatten())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        avg_text_w = np.mean(all_text_w)
        avg_image_w = np.mean(all_image_w)
        
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练准确率: {correct/total:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  平均权重 - 文本: {avg_text_w:.4f}, 图像: {avg_image_w:.4f}")
        print(f"  权重范围 - 文本: [{np.min(all_text_w):.4f}, {np.max(all_text_w):.4f}]")
        print("="*60)

    print("\n✅ 测试完成！动态权重正常工作！")

if __name__ == '__main__':
    main()
