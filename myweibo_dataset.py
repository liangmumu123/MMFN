import os
import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import clip

# 获取当前文件所在目录（即项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
clipmodel, preprocess = clip.load('ViT-B/32', device)
for param in clipmodel.parameters():
    param.requires_grad = False

# 使用 AutoFeatureExtractor 加载 Swin 特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "./data/swin-base-patch4-window7-224", local_files_only=True
)
# 加载 BERT tokenizer
token = BertTokenizer.from_pretrained('./data/bert-chinese', local_files_only=True)


def read_img(imgs, root_path, LABLEF):
    """
    从图像路径列表中随机选择一张图片，并返回其数组和 PIL 图像。
    imgs: 图像文件名列表（可能包含路径前缀）
    root_path: 数据集根目录（weibo 文件夹）
    LABLEF: 子文件夹名（rumor_images 或 nonrumor_images）
    """
    # 随机选择一张图片（此处随机，实际可按需修改）
    GT_path = imgs[np.random.randint(0, len(imgs))]
    # 仅取文件名（去除可能的前缀路径）
    GT_path = GT_path.split('/')[-1]
    # 拼接完整路径
    GT_path = os.path.join(root_path, LABLEF, GT_path)

    try:
        img_GT = util.read_img(GT_path)           # 返回 numpy 数组
        img_pro = Image.open(GT_path).convert('RGB')
    except Exception as e:
        img_GT = np.zeros((224, 224, 3))
        img_pro = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')
        print("找不到图片，使用默认图:", GT_path)
    return img_GT, img_pro


class weibo_dataset(data.Dataset):
    def __init__(self, is_train=True):
        super(weibo_dataset, self).__init__()
        self.label_dict = []
        self.swin = feature_extractor
        self.preprocess = preprocess
        # 数据集根目录：当前文件所在目录下的 weibo 文件夹
        self.local_path = os.path.join(current_dir, "weibo")

        csv_path = os.path.join(
            self.local_path,
            '{}_weibo_final3.csv'.format('train' if is_train else 'test')
        )
        gc = pandas.read_csv(csv_path)

        for i in tqdm(range(len(gc))):
            images_name = str(gc.iloc[i, 1])
            label = int(gc.iloc[i, 2])
            content = str(gc.iloc[i, 4])
            sum_content = str(gc.iloc[i, 4])
            has_image = gc.iloc[i, 6]
            record = {
                'images': images_name,
                'label': label,
                'content': content,
                'sum_content': sum_content,
                'has_image': has_image
            }
            self.label_dict.append(record)

    def __getitem__(self, item):
        record = self.label_dict[item]
        images, label, content, sum_content, has_image = (
            record['images'],
            record['label'],
            record['content'],
            record['sum_content'],
            record['has_image']
        )

        if label == 0:
            LABLEF = 'rumor_images'
        else:
            LABLEF = 'nonrumor_images'

        imgs = images.split('|')
        if has_image:
            img_GT, img_pro = read_img(imgs, self.local_path, LABLEF)
        else:
            img_GT = np.zeros((224, 224, 3))
            img_pro = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')

        sent = content
        image_swin = self.swin(img_GT, return_tensors="pt").pixel_values
        image_clip = self.preprocess(img_pro)
        text_clip = sum_content
        return (sent, image_swin, image_clip, text_clip), label

    def __len__(self):
        return len(self.label_dict)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


def collate_fn(data):
    """
    自定义批处理函数，用于 DataLoader。
    """
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    imageclip = [i[0][2] for i in data]
    textclip = [i[0][3] for i in data]
    labels = [i[1] for i in data]

    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding='max_length',
        max_length=300,
        return_tensors='pt',
        return_length=True
    )

    textclip = clip.tokenize(textclip, truncate=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    image = torch.stack(image).squeeze(1)
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels


# 测试代码：验证数据集能否正常加载
if __name__ == "__main__":
    print("当前文件所在目录:", current_dir)
    dataset = weibo_dataset(is_train=True)
    print("数据集根目录:", dataset.local_path)
    print("CSV 文件存在:", os.path.exists(os.path.join(dataset.local_path, 'train_weibo_final3.csv')))
    print("rumor_images 存在:", os.path.exists(os.path.join(dataset.local_path, 'rumor_images')))
    print("nonrumor_images 存在:", os.path.exists(os.path.join(dataset.local_path, 'nonrumor_images')))

    # 尝试获取一个样本
    try:
        sample, label = dataset[0]
        print("样本加载成功，标签:", label)
        # 打印样本信息
        sent, image_swin, image_clip, text_clip = sample
        print("文本内容长度:", len(sent))
        print("Swin 特征形状:", image_swin.shape)
        print("CLIP 图像特征形状:", image_clip.shape)
        print("CLIP 文本 token 数量:", len(text_clip))
    except Exception as e:
        print("样本加载失败:", e)