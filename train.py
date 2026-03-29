import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from transformers import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Core import MultiModal
from tqdm import tqdm
from myweibo_dataset import *
# from gossipcop_dataset import *
# from twitter_dataset import *

# Set logging verbosity to warning and error levels for transformers
logging.set_verbosity_warning()
logging.set_verbosity_error()

# Set CUDA_VISIBLE_DEVICES to control GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if CUDA is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# Define the training function
def train():
    batch_size = 8
    patience = 5  # 早停机制的耐心参数，如果验证集损失在连续5个epoch没有改善，就停止训练
    best_loss = np.inf
    patience_counter = 0  # 用于早停机制

    # Load training and validation datasets
    # train_set = twitter_dataset(is_train=True)
    # validate_set = twitter_dataset(is_train=False)
    train_set = weibo_dataset(is_train=True)
    validate_set = weibo_dataset(is_train=False)
    # train_set = gossipcop_dataset(is_train=True)
    # validate_set = gossipcop_dataset(is_train=False)

    # Create data loaders for training and testing
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,          # 改为0，避免多进程
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=False        # 可加
    )

    test_loader = DataLoader(
        validate_set,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=False
    )

    # Initialize the MultiModal model
    rumor_module = MultiModal()
    # rumor_module.forward = rumor_module.forward_no_image
    rumor_module.to(device)

    # Define the CrossEntropyLoss criterion for rumor classification
    loss_f_rumor = torch.nn.CrossEntropyLoss()

    # Extract parameters for optimizer groups
    # 提取 BERT 和 Swin 模型的参数 ID，用于区分不同参数组
    base_params = list(map(id, rumor_module.bert.parameters()))
    base_params += list(map(id, rumor_module.swin.parameters()))

    # Define the optimizer with different learning rates for different parameter groups
    # 使用 Adam 优化器
    optim_task = torch.optim.Adam([
        # 过滤掉 BERT 和 Swin 模型的参数，对其他参数使用默认学习率
        {'params': filter(lambda p: p.requires_grad and id(p) not in base_params, rumor_module.parameters())},
        {'params': rumor_module.bert.parameters(), 'lr': 1e-5},
        {'params': rumor_module.swin.parameters(), 'lr': 1e-5}
    ], lr=1e-3)
    # 对 BERT 和 Swin 模型设置为 1e-5，对其他参数设置为 1e-3

    # Training loop
    for epoch in range(50):  # 假设训练最多50个epoch
        print("start to train")
        rumor_module.train()  # 设置训练模式
        # 用于计算训练集的准确率和损失
        corrects_pre_rumor = 0  # 用于统计正确预测的样本数
        loss_total = 0  # 用于累加总损失
        rumor_count = 0  # 用于统计总样本数
        tk0 = tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)  # 创建一个带有进度条的迭代器
        # smoothing=0: 不对进度进行平滑处理。
        # mininterval=1.0: 设置最小更新间隔为 1.0 秒
        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(tk0):
            # 遍历训练数据加载器 tk0，获取批次索引 i 和批次数据
            # Transfer data to the appropriate device
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = to_var(input_ids), to_var(
                attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(
                label)

            # Encode image and text data using pre-trained CLIP models
            # 禁用梯度计算，用于编码图像和文本数据
            with torch.no_grad():
                #  使用预训练的 CLIP 模型编码图像和文本数据
                image_clip = clipmodel.encode_image(imageclip)
                text_clip = clipmodel.encode_text(textclip)

            # Forward pass through the MultiModal model
            pre_rumor = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip, image_clip)

            # Calculate the rumor loss
            loss_rumor = loss_f_rumor(pre_rumor, label)
            # 计算预测结果和真实标签之间的损失

            # Backpropagation and optimization
            optim_task.zero_grad()  # 清空优化器的梯度
            loss_rumor.backward()  # 计算损失的反向传播，得到梯度
            optim_task.step()  # 更新模型参数

            # Calculate accuracy and update counters
            pre_label_rumor = pre_rumor.argmax(1)  # 获取预测标签
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()  # 统计正确预测的样本数
            loss_total += loss_rumor.item() * input_ids.shape[0]  # 累加总损失
            rumor_count += input_ids.shape[0]  # 统计总样本数

        # Calculate training accuracy and loss
        loss_rumor_train = loss_total / rumor_count  # 计算训练集的平均损失
        acc_rumor_train = corrects_pre_rumor / rumor_count  # 计算训练集的准确率

        # Evaluate on the test set
        acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor = test(
            rumor_module, test_loader)

        # Print results
        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f || loss_rumor_train = %.3f || loss_rumor_test = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test, loss_rumor_train, loss_rumor_test))

        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)

        # 早停机制
        if loss_rumor_test < best_loss:
            best_loss = loss_rumor_test  # 记录最佳损失
            patience_counter = 0  # 记录连续未改善的 epoch 数
            torch.save(rumor_module.state_dict(), 'best_model.pth')  # 保存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # 加载最佳模型的参数
    rumor_module.load_state_dict(torch.load('best_model.pth'))
    return rumor_module, test_loader


# 将输入数据转移到 GPU（如果可用）并包装为 Variable，以便进行自动求导。
# Helper function to transfer data to the appropriate device
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# 将模型设置为评估模式，禁用 Dropout 和 Batch Normalization 等层的训练行为。
# Helper function to test the model
def test(rumor_module, test_loader):
    rumor_module.eval()

    loss_f_rumor = torch.nn.CrossEntropyLoss()

    rumor_count = 0
    loss_total = 0
    # 用于收集真实标签和预测标签
    rumor_label_all = []
    rumor_pre_label_all = []

    # 禁用梯度计算，减少内存消耗并加速计算
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(test_loader):
            # Transfer data to the appropriate device
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = to_var(input_ids), to_var(
                attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(
                label)

            # Encode image and text data using pre-trained CLIP models
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(textclip)

            # Forward pass through the MultiModal model
            pre_rumor = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip, image_clip)
            loss_rumor = loss_f_rumor(pre_rumor, label)
            pre_label_rumor = pre_rumor.argmax(1)

            loss_total += loss_rumor.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

            # Collect predictions and labels
            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(label.detach().cpu().numpy())

        # Calculate metrics
        loss_rumor_test = loss_total / rumor_count
        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)

        acc_rumor_test = accuracy_score(rumor_label_all, rumor_pre_label_all)
        precision_rumor_test = precision_score(rumor_label_all, rumor_pre_label_all, average=None)
        recall_rumor_test = recall_score(rumor_label_all, rumor_pre_label_all, average=None)
        f1_rumor_test = f1_score(rumor_label_all, rumor_pre_label_all, average=None)
        conf_rumor = confusion_matrix(rumor_label_all, rumor_pre_label_all)

        # Generate classification report
        classification_report_rumor = classification_report(rumor_label_all, rumor_pre_label_all,
                                                            target_names=['realnews', 'fakenews'], digits=4)

    print("Overall Accuracy:", acc_rumor_test)
    print("Precision per class:", precision_rumor_test)
    print("Recall per class:", recall_rumor_test)
    print("F1 Score per class:", f1_rumor_test)
    print("Confusion Matrix:\n", conf_rumor)
    print("Classification Report:\n", classification_report_rumor)

    return acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor


# Entry point
if __name__ == "__main__":
    model, test_loader = train()
    test(model, test_loader)
