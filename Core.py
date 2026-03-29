from random import random
import torch
import torch.nn as nn
import math
import random
import torch.backends.cudnn as cudnn
import numpy as np
import copy
from transformers import BertConfig, BertModel, SwinModel

# Set a manual seed for reproducibility 设置随机种子以确保实验的可重复性
manualseed = 666
random.seed(manualseed)  # 设置 Python 的随机数生成器的种子
np.random.seed(manualseed)  # 设置 NumPy 的随机数生成器的种子
torch.manual_seed(manualseed)  # 设置 PyTorch 的随机数生成器的种子
torch.cuda.manual_seed(manualseed)  # 设置 PyTorch 在 GPU 上的随机数生成器的种子
cudnn.deterministic = True  # 确保 CuDNN（NVIDIA 的深度神经网络库）使用确定性算法

# Load BERT model and configure its output
# model_name = './bert-base-chinese'
model_name = "bert-base-chinese"
config = BertConfig.from_pretrained("./data/bert-chinese", num_labels=2)
config.output_hidden_states = False


# Definition of the Transformer model
class Transformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, number_of_layers, dropout_probability,
                 log_attention_weights=False):
        super().__init__()
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha)
        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text, image):
        src_representations_batch1 = self.encode(text, image)
        src_representations_batch2 = self.encode(image, text)
        return src_representations_batch1, src_representations_batch2

    def encode(self, src1, src2):
        src_representations_batch = self.encoder(src1, src2)
        return src_representations_batch


class Encoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src1, src2):
        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src1, src2)
        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, dropout_probability, multi_headed_attention):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)
        self.multi_headed_attention = multi_headed_attention
        self.model_dimension = model_dimension

    def forward(self, srb1, srb2):
        encoder_self_attention = lambda srb1, srb2: self.multi_headed_attention(query=srb1, key=srb2, value=srb2)
        src_representations_batch = self.sublayers[0](srb1, srb2, encoder_self_attention)
        return src_representations_batch


class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, srb1, srb2, sublayer_module):
        return srb1 + self.dropout(sublayer_module(self.norm(srb1), self.norm(srb2)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.softmax = nn.Softmax(dim=-1)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        return intermediate_token_representations, attention_weights

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        if self.log_attention_weights:
            self.attention_weights = attention_weights

        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)

        token_representations = self.out_projection_net(reshaped)
        return token_representations


def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    print(model.state_dict().keys())
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=256, prime_dim=16, pre_dim=2):
        super(UnimodalDetection, self).__init__()

        self.text_uni = nn.Sequential(
            nn.Linear(1280, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU())

        self.image_uni = nn.Sequential(
            nn.Linear(1536, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU())

    def forward(self, text_encoding, image_encoding):
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class CrossModule(nn.Module):
    def __init__(self, corre_out_dim=16):
        super(CrossModule, self).__init__()
        self.corre_dim = 1024
        self.c_specific_1 = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.c_specific_3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image, text1, image1):
        correlation_out = self.c_specific_1(torch.cat((text, image), 1).float())
        correlation_out1 = self.c_specific_2(torch.cat((text1, image1), 1).float())
        correlation_out2 = self.c_specific_3(torch.cat((correlation_out, correlation_out1), 1))
        return correlation_out2


class MultiModal(nn.Module):
    def __init__(self, feature_dim=48, h_dim=48):
        super(MultiModal, self).__init__()

        # Initialize learnable parameters
        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

        # Initialize the Transformer model for cross-modal attention
        self.trans = Transformer(model_dimension=512, number_of_heads=8, number_of_layers=1, dropout_probability=0.1,
                                 log_attention_weights=False)

        self.t_projection_net = nn.Linear(768, 512)
        self.i_projection_net = nn.Linear(1024, 512)

        # Load the Swin Transformer model for image processing
        self.swin = SwinModel.from_pretrained("./data/swin-base-patch4-window7-224", local_files_only=True).to("cpu")
        for param in self.swin.parameters():
            param.requires_grad = True

        # Load BERT model for text processing
        self.bert = BertModel.from_pretrained("./data/bert-chinese", local_files_only=True)
        for param in self.bert.parameters():
            param.requires_grad = True

        # Initialize unimodal representation modules
        self.uni_repre = UnimodalDetection()

        # Initialize cross-modal fusion module
        self.cross_module = CrossModule()

        # ========== 新增：细粒度对齐投影层 ==========
        self.fine_align_proj = nn.Linear(512, 16)

        # ========== 保留原有的分类器（48维）用于消融实验 ==========
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )

        # ========== 新增主分类器（64维）用于多粒度模型 ==========
        self.classifier_main = nn.Sequential(
            nn.Linear(64, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )

    # 以下 forward_* 方法保留原样，使用 self.classifier_corre（48维）
    def forward_no_unimodal(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300,
                                        torch.sum(image_att, dim=1) / 49)
        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(
                            torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        correlation = correlation * mweight
        final_feature = torch.cat([correlation, correlation, correlation], 1)
        pre_label = self.classifier_corre(final_feature)
        return pre_label

    def forward_no_image(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)
        text_prime, _ = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        sim = torch.div(torch.sum(text * text, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(text, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        final_feature = torch.cat([text_prime, text_prime, text_prime], 1)
        pre_label = self.classifier_corre(final_feature)
        return pre_label

    def forward_no_text(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)
        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        sim = torch.div(torch.sum(text * text, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(text, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        final_feature = torch.cat([image_prime, image_prime, image_prime], 1)
        pre_label = self.classifier_corre(final_feature)
        return pre_label

    def forward_no_clip(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)
        text = torch.ones_like(text)
        image = torch.ones_like(image)
        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
        final_feature = torch.cat([text_prime, image_prime, correlation], 1)
        pre_label = self.classifier_corre(final_feature)
        return pre_label

    def forward_no_transformer(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        correlation = self.cross_module(text, image, text_m, image_m)
        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        correlation = correlation * mweight
        final_feature = torch.cat([text_prime, image_prime, correlation], 1)
        pre_label = self.classifier_corre(final_feature)
        return pre_label

    def forward_no_weight(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)

        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
        final_feature = torch.cat([text_prime, image_prime, correlation], 1)
        pre_label = self.classifier_corre(final_feature)
        return pre_label

    def forward_no_crossmodule(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        image_raw = self.swin(image_raw)

        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw.pooler_output, image.flatten(1)], 1))
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)
        text_att, image_att = self.trans(text_m, image_m)
        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        final_feature = torch.cat([text_prime, image_prime, text_prime], 1)
        pre_label = self.classifier_corre(final_feature)
        return pre_label

    # ========== 主 forward，使用多粒度对齐模块和 classifier_main ==========
    def forward(self, input_ids, attention_mask, token_type_ids, image_raw, text, image):
        # Extract features using BERT for textual input
        BERT_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_states = BERT_feature['last_hidden_state']

        # Compute raw text feature by averaging over tokens
        text_raw = torch.sum(last_hidden_states, dim=1) / 300
        # Process the raw image feature using Swin Transformer
        image_raw = self.swin(image_raw)

        # Generate unimodal representations for text and image
        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1),
                                                 torch.cat([image_raw.pooler_output, image.flatten(1)], 1))

        # Project text and image features to a common space
        text_m = self.t_projection_net(last_hidden_states)   # [b, seq_len, 512]
        image_m = self.i_projection_net(image_raw.last_hidden_state)   # [b, num_patches, 512]

        # Apply cross-modal attention via Transformer
        text_att, image_att = self.trans(text_m, image_m)

        # Cross-modal fusion using the cross-module
        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)

        # Compute CLIP similarity between text and image features
        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)
        correlation = correlation * mweight

        # ========== 多粒度对齐模块 ==========
        # text_m: [batch, seq_len, 512]
        # image_m: [batch, num_patches, 512]   # num_patches = 49
        fine_grained_sim = torch.matmul(text_m, image_m.transpose(1, 2)) / math.sqrt(512)  # [b, seq_len, num_patches]
        att_weights = torch.softmax(fine_grained_sim, dim=-1)  # [b, seq_len, num_patches]
        fine_grained_image = torch.matmul(att_weights, image_m)  # [b, seq_len, 512]
        fine_grained_align = torch.mean(fine_grained_image, dim=1)  # [b, 512]
        fine_align_proj = self.fine_align_proj(fine_grained_align)  # [b, 16]

        # Combine all features for final prediction
        final_feature = torch.cat([text_prime, image_prime, correlation, fine_align_proj], 1)  # [b, 64]

        # final prediction using the new classifier
        pre_label = self.classifier_main(final_feature)

        return pre_label