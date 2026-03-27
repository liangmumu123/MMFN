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
config = BertConfig.from_pretrained(model_name, num_labels=2)  # 加载 BERT 模型的配置
config.output_hidden_states = False

# 多头注意力
class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att


    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


# 实现跨模态注意力机制，处理文本和图像的交互
class Transformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, number_of_layers, dropout_probability,
                 log_attention_weights=False):
        # __init__ 是类的构造函数，接收以下参数：
        # model_dimension: 模型的维度（嵌入维度）。
        # number_of_heads: 多头注意力机制中的头数。
        # number_of_layers: 编码器层的数量。
        # dropout_probability: Dropout 的概率，用于防止过拟合。
        # log_attention_weights: 是否记录注意力权重，默认为 False。
        super().__init__()
        # All of these will get deep-copied multiple times internally
        # 创建一个多头注意力（Multi-Head Attention）模块 mha，用于计算注意力权重
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        # 创建一个编码器层 encoder_layer，包含多头注意力和前馈网络
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha)
        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.init_params()

    # 初始化模型参数
    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, text, image):
        src_representations_batch1 = self.encode(text, image)  # 对文本和图像进行编码，返回编码后的表示
        src_representations_batch2 = self.encode(image, text)  # 对图像和文本进行编码，返回编码后的表示
        return src_representations_batch1, src_representations_batch2
       # 改：  return text,image


    def encode(self, src1, src2):
        src_representations_batch = self.encoder(src1, src2)  # forward pass through the encoder
        return src_representations_batch


# 编码器相关类
class Encoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layers = get_clones(encoder_layer, number_of_layers)  # 使用 get_clones 函数复制多个 EncoderLayer 对象，形成编码器层堆叠。
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)  # 用于对输出进行归一化

    def forward(self, src1, src2):
        # Forward pass through the encoder stack
        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src1, src2)
        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention):
        super().__init__()
        num_of_sublayers_encoder = 2  # 定义编码器层中的子层数量为2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention

        self.model_dimension = model_dimension

    def forward(self, srb1, srb2):
        encoder_self_attention = lambda srb1, srb2: self.multi_headed_attention(query=srb1, key=srb2, value=srb2)
        # 定义一个 lambda 函数，表示编码器的自注意力机制

        src_representations_batch = self.sublayers[0](srb1, srb2, encoder_self_attention)
        return src_representations_batch


class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)  # 层归一化
        self.dropout = nn.Dropout(p=dropout_probability)  # Dropout层，防止过拟合

    def forward(self, srb1, srb2, sublayer_module):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return srb1 + self.dropout(sublayer_module(self.norm(srb1), self.norm(srb2)))
   # 使用残差连接：将输入 srb1 与子层模块的输出相加。
   # 子层模块的输入经过 LayerNorm 和 Dropout 处理。
   # 返回残差连接的结果。

# 实现多头自注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        # 使用 get_clones 函数创建三个线性层，分别用于生成查询（Query）、键（Key）和值（Value）
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)
        # 创建一个线性层，用于将多头注意力的输出投影到模型维度

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension
        # 创建一个 Softmax 层，用于计算注意力权重

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        # 计算查询和键的点积，然后除以头维度的平方根
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        # 将注意力权重应用到值向量上，得到中间表示

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, ):
        batch_size = query.shape[0]

        # 将查询、键和值通过线性层转换，并调整维度以适应多头注意力
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        # 调用 attention 方法计算注意力权重和中间表示
        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        if self.log_attention_weights:
            self.attention_weights = attention_weights
        # 将中间表示重新调整维度，并通过输出投影层得到最终表示。
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1,
                                                                              self.number_of_heads * self.head_dimension)

        token_representations = self.out_projection_net(reshaped)

        return token_representations


# Utility function to create deep copies of a module
# 创建模块的深度拷贝
def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


# 统计模型的可训练参数数量
# Function to count trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 分析模型的状态字典
# Function to analyze the shapes and names of parameters in a state dict
def analyze_state_dict_shapes_and_names(model):
    print(model.state_dict().keys())

    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


# 处理单一模态（文本或图像）的特征提取。
# Definition of the Unimodal Detection model
class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=256, prime_dim=16, pre_dim=2):
        # 接收三个参数：
        # shared_dim: 共享维度，默认为 256。
        # prime_dim: 主维度，默认为 16。
        # pre_dim: 预测维度，默认为 2。
        super(UnimodalDetection, self).__init__()

        self.text_uni = nn.Sequential(
            nn.Linear(1280, shared_dim),  # 将输入维度 1280 映射到共享维度 256
            nn.BatchNorm1d(shared_dim),  # 对共享维度进行批量归一化
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(shared_dim, prime_dim),  # 将共享维度映射到主维度 16
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
        text_prime = self.text_uni(text_encoding)  # 通过文本单一模态网络提取的特征
        image_prime = self.image_uni(image_encoding)  # 通过图像单一模态网络提取的特征
        return text_prime, image_prime


# 用于跨模态特征的交互和融合
# Definition of the Cross-Modal model
class CrossModule(nn.Module):
    def __init__(
            self,
            corre_out_dim=16):  # 输出维度，默认为 16
        super(CrossModule, self).__init__()
        self.corre_dim = 1024  # 定义跨模态维度为 1024
        # 定义第一个跨模态特定网络
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
        correlation_out = self.c_specific_1(torch.cat((text, image), 1).float())  #  将 text 和 image 拼接后通过第一个跨模态特定网络
        correlation_out1 = self.c_specific_2(torch.cat((text1, image1), 1).float())  # 将 text1 和 image1 拼接后通过第二个跨模态特定网络
        correlation_out2 = self.c_specific_3(torch.cat((correlation_out, correlation_out1), 1))  # 将 correlation_out 和 correlation_out1 拼接后通过第三个跨模态特定网络。
        return correlation_out2





# Definition of the MultiModal model
class MultiModal(nn.Module):
    # 接收两个参数：
    # feature_dim: 特征维度，默认为 48。
    # h_dim: 隐藏层维度，默认为 48。
    def __init__(
            self,
            feature_dim=48,
            h_dim=48
    ):
        super(MultiModal, self).__init__()

        # Initialize learnable parameters
        self.w = nn.Parameter(torch.rand(1))  # Learnable parameter for weighting similarity
        self.b = nn.Parameter(torch.rand(1))  # Learnable parameter for biasing similarity

        # Initialize the TransformerBlock model for multi-head attention
        self.mh_attention = TransformerBlock(input_size=862, n_heads=8, attn_dropout=0)

        # Initialize the Transformer model for cross-modal attention
        self.trans = Transformer(model_dimension=512, number_of_heads=8, number_of_layers=1, dropout_probability=0.1,
                                 log_attention_weights=False)



        # Initialize the Transformer model for cross-modal attention
        # 文本特征投影网络，将 768 维度映射到 512 维度
        self.t_projection_net = nn.Linear(768, 512)  # Linear projection for text
        # 图像特征投影网络，将 1024 维度映射到 512 维度
        self.i_projection_net = nn.Linear(1024, 512)  # Linear projection for Image

        # Load the Swin Transformer model for image processing
        self.swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224").cuda()
        for param in self.swin.parameters():
            param.requires_grad = True  # 确保所有参数可训练

        # Load BERT model for text processing
        self.bert = BertModel.from_pretrained(model_name, config=config).cuda()
        for param in self.bert.parameters():
            param.requires_grad = True

        # Initialize unimodal representation modules
        self.uni_repre = UnimodalDetection()

        # Initialize cross-modal fusion module
        self.cross_module = CrossModule()

        # Define classifier layers for final prediction
        self.classifier_corre = nn.Sequential(
             nn.Linear(feature_dim, h_dim),  # 将输入维度 feature_dim 映射到隐藏层维度 h_dim
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
           nn.Linear(h_dim, 2)  # 将隐藏层维度映射到输出维度 2（二分类）
         )

        # Class  Classifier
        # self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)  # 将图像从原始维度映射到隐藏层维度
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        # # 单模态分类器
        # self.unimodal_classifier = nn.Sequential(
        #     nn.Linear(feature_dim, 64),  # 隐藏层维度为64
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2),
        #     nn.Softmax()
        # )
        #
        # # 融合模态分类器
        # self.fusion = nn.Sequential(
        #     nn.Linear(feature_dim * 2, feature_dim * 2),
        #     nn.BatchNorm1d(sim_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(sim_dim * 2, sim_dim),
        #     nn.ReLU()
        # )
        #
        # self.sim_classifier = nn.Sequential(
        #     nn.Linear(sim_dim * 3, sim_dim),
        #     nn.BatchNorm1d(sim_dim),
        #     nn.ReLU(),
        #     nn.Linear(sim_dim, 2)
        # )

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

        # 增强的单模态特征
        self_att_t = self.mh_attention(text_raw.view(bsz, -1, 862), text_raw.view(bsz, -1, 862), \
                                       text_raw.view(bsz, -1, 862))

        self_att_i = self.mh_attention(image_raw.view(bsz, -1, 862), image_raw.view(bsz, -1, 862), \
                                       image_raw.view(bsz, -1, 862))

        self_i = self_att_i.view(bsz, 862)
        self_t = self_att_t.view(bsz, 862)

        # 通过单一模态模块提取特征
        # Generate unimodal representations for text and image
        text_prime, image_prime = self.uni_repre(torch.cat([self_t, text], 1),
                                                 torch.cat([self_i, image], 1))


        # Project text and image features to a common space
        text_m = self.t_projection_net(last_hidden_states)
        image_m = self.i_projection_net(image_raw.last_hidden_state)

        #  通过 Transformer 进行跨模态注意力
        # Apply cross-modal attention
        text_att, image_att = self.trans(text_m, image_m)

        # Cross-modal fusion using the cross-module
        correlation = self.cross_module(text, image, torch.sum(text_att, dim=1) / 300, torch.sum(image_att, dim=1) / 49)
        # 对文本注意力特征 text_att 沿序列维度求和并平均池化,对图像注意力特征 image_att 沿图像块维度求和并平均池化

        # Compute CLIP similarity between text and image features
        sim = torch.div(torch.sum(text * image, 1),
                        torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))

        # Apply learned weighting and bias to similarity
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)

        # Weighted cross-modal fusion
        correlation = correlation * mweight

        # Combine all features for final prediction
        final_feature = torch.cat([text_prime, image_prime, correlation], 1)

        # final prediction
        pre_label = self.classifier_corre(final_feature)

        return pre_label
