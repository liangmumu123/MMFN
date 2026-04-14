
from transformers import AutoFeatureExtractor

print("尝试下载 Swin Transformer...")
try:
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")
    print("Swin 下载成功！")
except Exception as e:
    print("Swin 下载失败！")
    print("错误信息：")
    print(e)
