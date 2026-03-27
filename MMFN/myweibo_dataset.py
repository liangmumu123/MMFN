import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, ViTImageProcessor
import clip
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

clipmodel, preprocess = clip.load('ViT-B/32', device)
for param in clipmodel.parameters():
    param.requires_grad = False

feature_extractor = ViTImageProcessor.from_pretrained("./data/swin-base-patch4-window7-224", local_files_only=True)
token = BertTokenizer.from_pretrained('./data/bert-chinese', local_files_only=True)

def read_img(imgs, root_path, LABLEF):
    GT_path = imgs[np.random.randint(0, len(imgs))]
    GT_path = GT_path.split('/')[-1]
    GT_path = "D:/AIprojects/其他/NLP舆情/虚假新闻检测数据集/weibo/" + LABLEF + "/" + GT_path

    try:
        img_GT = util.read_img(GT_path)
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
        self.local_path = "D:/AIprojects/其他/NLP舆情/虚假新闻检测数据集/weibo"

        csv_path = self.local_path + '/{}_weibo_final3.csv'.format('train' if is_train else 'test')
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
        images, label, content, sum_content, has_image = record['images'], record['label'], record['content'], record['sum_content'], record['has_image']

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


# ✅ ✅ ✅ 【最终正确】collate_fn 独立函数
def collate_fn(data):
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

    # ✅【真正正确】适配你的 MMFN 模型！！！
    image = torch.stack(image).squeeze(1)
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels