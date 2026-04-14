import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, input_ids, attention_mask, token_type_ids, image_swin, image_clip=None, text_clip=None,
                labels=None, return_weights=False):
        text_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
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
            if return_weights:
                return loss, logits, text_weight, image_weight
            return loss, logits

        if return_weights:
            return logits, text_weight, image_weight
        return logits
