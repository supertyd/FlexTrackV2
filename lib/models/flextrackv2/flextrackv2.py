"""
FlexTrackV2 Model
"""
import random
import torch
import math
from torch import nn
import torch.nn.functional as F
from lib.models.flextrackv2.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .neck import build_neck
from collections import OrderedDict
from .moe_fusion import MoEFusion
import numpy as np

import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

def visulize_attention_ratio(img_path, attention_mask, ratio=0.5, cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    
    # normalize the attention mask
    # mask = cv2.resize(attention_mask, (img_h, img_w))
    mask = F.interpolate(attention_mask, size=(img_h, img_w), mode='bilinear', align_corners=False)
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255)
    # plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.imshow(normed_mask.cpu().numpy()[0][0], alpha=0.5, interpolation='bilinear', cmap="jet")
    plt.savefig("output.png", dpi=300, bbox_inches='tight')






def apply_mask(tensor1, tensor2, mask_list, rgb_prompt=None, aux_prompt=None):
    # 输入:
    #   tensor1: [32, 245, 512]
    #   tensor2: [32, 245, 512]
    #   mask_list: 32个元素的列表，每个元素是 [2,5]
    # 输出: 两个掩码后的Tensor

    # 转换 mask_list 为 tensor，并添加 batch 维度，得到 shape [32, 2, 5, 1, 1]

    batch_size = tensor1.shape[0]
    masks = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)  # [32, 2, 5, 1, 1]

    total_template_tokens = tensor1.shape[1]
    embed_dim = tensor1.shape[2]
    num_patches_template = total_template_tokens // 5

    # 将 tensor1 和 tensor2 转换为 [B, 5, num_patches_template, embed_dim] 的形状
    tensor1 = tensor1.view(batch_size, 5, num_patches_template, embed_dim)
    tensor2 = tensor2.view(batch_size, 5, num_patches_template, embed_dim)

    # 将 mask 广播并应用掩码
    rgb_mask = masks[:, 0].to(tensor1.device)  # 应用第0行的掩码
    aux_mask = masks[:, 1].to(tensor1.device)  # 应用第1行的掩码

    if rgb_prompt is not None and aux_prompt is not None:
        masked_tensor1 = tensor1 * rgb_mask + rgb_prompt.unsqueeze(1) * (1.0 - rgb_mask)
        masked_tensor2 = tensor2 * aux_mask + aux_prompt.unsqueeze(1) * (1.0 - aux_mask)
    else:
        masked_tensor1 = tensor1 * rgb_mask
        masked_tensor2 = tensor2 * aux_mask

    # 恢复形状为 [B, total_template_tokens, embed_dim]
    masked_tensor1 = masked_tensor1.view(batch_size, total_template_tokens, embed_dim)
    masked_tensor2 = masked_tensor2.view(batch_size, total_template_tokens, embed_dim)

    return masked_tensor1, masked_tensor2

def random_mask(tensor, mask_prob, num_tokens_to_mask):
    """
    对输入的 Tensor 进行随机 mask。
    
    参数:
        tensor: 输入的 Tensor,形状为 [batch_size, seq_len, hidden_dim]。
        mask_prob: 每个 batch 被 mask 的概率。
        num_tokens_to_mask: 如果被 mask,每个 batch 中需要 mask 的 token 数量。
    
    返回:
        masked_tensor: 经过 mask 后的 Tensor。
        mask: 生成的 mask,形状与 tensor 相同。
    """
    batch_size, seq_len, hidden_dim = tensor.shape
    
    # 1. 决定每个 batch 是否被 mask
    batch_mask = torch.rand(batch_size) < mask_prob  # 形状: [batch_size]
    
    # 2. 生成 token 级别的 mask
    token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)  # 形状: [batch_size, seq_len]
     
    for i in range(batch_size):
        if batch_mask[i]:  # 如果该 batch 被 mask
            # 随机选择 num_tokens_to_mask 个 token 进行 mask
            # num_tokens_to_mask = random.randint(122, 245)

            indices = torch.randperm(seq_len)[:num_tokens_to_mask]
            token_mask[i, indices] = True
    
    # 3. 将 token_mask 扩展到与 tensor 相同的形状
    token_mask = token_mask.unsqueeze(-1).expand_as(tensor)  # 形状: [batch_size, seq_len, hidden_dim]
    
    # 4. 应用 mask
    masked_tensor = tensor.masked_fill(token_mask.to(tensor.device), 0)  # 将 mask 为 True 的位置置为 0
    
    return masked_tensor, token_mask



# 示例输入
# tensor1 = torch.randn(32, 192, 512)
# tensor2 = torch.randn(32, 192, 512)

# # 对 tensor1 和 tensor2 分别进行 mask
# masked_tensor1, mask1 = random_mask(tensor1, mask_prob=0.5, num_tokens_to_mask=60)
# masked_tensor2, mask2 = random_mask(tensor2, mask_prob=0.5, num_tokens_to_mask=60)

# # 打印结果
# print("Masked Tensor 1 shape:", masked_tensor1.shape)
# print("Masked Tensor 2 shape:", masked_tensor2.shape)




class LinearAttention_moe(nn.Module):
    def __init__(self, embed_size, hidden_size, patch_num_x, patch_num_z, moe_type=None, moe_cfg=None):
        super(LinearAttention_moe, self).__init__()
        self.value_proj = MoEFusion(input_size=embed_size, patch_num_x=patch_num_x, patch_num_z=patch_num_z, moe_type=moe_type, moe_cfg=moe_cfg)
        embed_size = embed_size*2
        hidden_size = hidden_size*2
        self.query_proj = nn.Linear(embed_size, hidden_size)
        self.key_proj = nn.Linear(embed_size, hidden_size)
        self.g_proj = nn.Linear(embed_size, hidden_size)

        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, int(embed_size/2))

        

    def forward(self, x, missing_mask=None):
        # Linear projections
        Q = self.silu(self.query_proj(x))  # Query
        K = self.silu(self.key_proj(x))    # Key

        # V = self.silu(self.value_proj(x))  # Value
        V, balance_loss = self.value_proj(x, missing_mask=missing_mask)
        V = self.silu(V)
        G = F.softmax(self.g_proj(x),dim=-1)  # Gate (softmax activation
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        # Attention mechanism (element-wise multiplication)
        attention = torch.einsum('bld,bmd->blm', Q, K)  # Q*K^T (linear attention)
        # attention = self.rmsnorm(attention)  # Apply RMSNorm to the attention

        # Weighted sum of values
        output = torch.einsum('blm,bmd->bld', attention, V)  # Attention weighted values

        # Apply the gate
        output = output * G

        # Final linear transformation
        output = self.linear(output)

        return output, balance_loss


def generate_random_matrix():
    matrix = [[random.choice([0, 1]) for _ in range(5)] for _ in range(2)]
    for col in range(5):
        if not any(matrix[row][col] == 1 for row in range(2)):
            matrix[random.randint(0, 1)][col] = 1
    return matrix


class LinearAttention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LinearAttention, self).__init__()
        embed_size = embed_size*2
        hidden_size = hidden_size*2
        self.query_proj = nn.Linear(embed_size, hidden_size)
        self.key_proj = nn.Linear(embed_size, hidden_size)
        self.value_proj = nn.Linear(embed_size, hidden_size)
        self.g_proj = nn.Linear(embed_size, hidden_size)

        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        # Linear projections
        Q = self.silu(self.query_proj(x))  # Query
        K = self.silu(self.key_proj(x))    # Key
        V = self.silu(self.value_proj(x))  # Value
        G = F.softmax(self.g_proj(x),dim=-1)  # Gate (softmax activation
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        # Attention mechanism (element-wise multiplication)
        attention = torch.einsum('bld,bmd->blm', Q, K)  # Q*K^T (linear attention)
        # attention = self.rmsnorm(attention)  # Apply RMSNorm to the attention
        # Weighted sum of values
        output = torch.einsum('blm,bmd->bld', attention, V)  # Attention weighted values
        # Apply the gate
        output = output * G
        # Final linear transformation
        output = self.linear(output)
        return output





class FlexTrackV2(nn.Module):
    """ This is the base class for FlexTrackV2 """
    def __init__(self, encoder, decoder, neck,cfg,
                 num_frames=1, num_template=1, decoder_type="CENTER"):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.neck = neck

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        self.freeze_en = cfg.TRAIN.FREEZE_ENCODER
        self.interaction_indexes = cfg.MODEL.ENCODER.INTERACTION_INDEXES
        embed_dim = self.encoder.num_channels
        self.gated_fusion = LinearAttention_moe(embed_dim, embed_dim, self.num_patch_x, self.num_patch_z, cfg.MODEL.MOE.TYPE, moe_cfg=cfg.MODEL.MOE)
        print('=== DEBUG SHAPES:', self.num_patch_x, self.num_patch_z, self.encoder.body.num_patches_search, '===')

        self.cfg = cfg

        # Learnable missing modality prompts (Scheme A)
        self.rgb_missing_prompt = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.aux_missing_prompt = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.rgb_missing_prompt, std=0.02)
        nn.init.trunc_normal_(self.aux_missing_prompt, std=0.02)

        # Context-Aware Dynamic Prompt Gating (CADP-Gating) for v13
        self.prompt_context_mapper = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, template_list=None, search_list=None, template_anno_list=None,enc_opt=None,neck_h_state=None, feature=None, mode="encoder",loss=None,type=None,missing = None, epoch=None):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder" and type == "dual":
            if self.training:
                xz, loss, xz_fusion_teacher = self.forward_encoder(template_list, search_list, template_anno_list, missing=missing, epoch=epoch)
                self.loss = loss
                return xz, loss, xz_fusion_teacher
            else:
                xz, loss = self.forward_encoder(template_list, search_list, template_anno_list, missing=missing, epoch=epoch)
                self.loss = loss
                return xz, loss
        if mode == "encoder" and type == "single":
            xz, loss = self.forward_encoder_single(template_list, search_list, template_anno_list)
            self.loss = loss
            return xz,loss

        
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state)
        elif mode == "decoder":
            

            return self.forward_decoder(feature,self.loss)
        else:
            raise ValueError

    def forward_encoder_single(self, template_list, search_list, template_anno_list):
        # Forward the encoder

        
        

        
        xz = self.encoder(template_list, search_list, template_anno_list)

        xz_aux = self.encoder(template_list_aux, search_list_aux, template_anno_list)


 




        xz_fusion, loss_balance = self.gated_fusion(torch.concat((xz,xz_aux),dim=-1))

        # xz_fusion,loss_balance = self.moefusion(xz_fusion)
        
    def forward_encoder(self, template_list, search_list, template_anno_list, missing = None, epoch=None, current_missing=None):
        # Forward the encoder
        template_list_rgb = [tensor[:,:3, :, :] for tensor in template_list]
        search_list_rgb = [tensor[:,:3, :, :] for tensor in search_list]
        template_list_aux = [tensor[:,3:, :, :] for tensor in template_list]
        search_list_aux = [tensor[:,3:, :, :] for tensor in search_list]
        
        
        # if not self.training:
        #     if missing == [1,1]:
        #         xz = self.encoder(template_list_rgb, search_list_rgb, template_anno_list)
        #         xz_aux = self.encoder(template_list_aux, search_list_aux, template_anno_list)
        #     elif missing == [0,1]:
        #         xz = self.encoder(template_list_rgb, [search_list_rgb[0]*0], template_anno_list)
        #         xz_aux = self.encoder(template_list_aux, search_list_aux, template_anno_list)
        #     elif missing == [1,0]:
        #         xz = self.encoder(template_list_rgb, search_list_rgb, template_anno_list)
        #         xz_aux = self.encoder(template_list_aux, [search_list_aux[0]*0], template_anno_list)
        # else:
        xz = self.encoder(template_list_rgb, search_list_rgb, template_anno_list)
        xz_aux = self.encoder(template_list_aux, search_list_aux, template_anno_list)


        if self.training:
            # Teacher Stream (Complete Modality)
            xz_fusion_teacher, loss_balance_teacher = self.gated_fusion(torch.concat((xz, xz_aux), dim=-1), missing_mask=torch.ones(xz.shape[0], 2, device=xz.device))
            xz_fusion_teacher = xz + xz_aux + xz_fusion_teacher

            # Student Stream (Curriculum Missing Augmentation)
            # Calculate p_miss based on epoch (linear scale from 10% to 50%)
            if epoch is not None:
                p_min = self.cfg.TRAIN.CMA_P_MIN
                p_max = self.cfg.TRAIN.CMA_P_MAX
                p_miss = p_min + min(1.0, max(0.0, (epoch - 1) / 39.0)) * (p_max - p_min)
            else:
                p_miss = (self.cfg.TRAIN.CMA_P_MIN + self.cfg.TRAIN.CMA_P_MAX) / 2.0

            batch_size = xz.shape[0]
            xz_student = xz.clone()
            xz_aux_student = xz_aux.clone()

            # Dynamic masking for search patches based on p_miss
            mask_data = []
            for _ in range(batch_size):
                if random.random() < p_miss:
                    if random.random() < 0.5:
                        mask_data.append([0.0, 1.0]) # drop RGB
                    else:
                        mask_data.append([1.0, 0.0]) # drop Aux
                else:
                    mask_data.append([1.0, 1.0])
            
            data = torch.tensor(mask_data, dtype=torch.float32, device=xz.device)

            rgb_mask = data[:, 0].unsqueeze(-1).unsqueeze(-1)
            aux_mask = data[:, 1].unsqueeze(-1).unsqueeze(-1)

            if self.cfg.MODEL.MOE.TYPE == "CADP_GATING":
                # Compute global average pooling context from active modality
                rgb_context = self.prompt_context_mapper(xz_aux.mean(dim=1, keepdim=True)) # [B, 1, 512]
                aux_context = self.prompt_context_mapper(xz.mean(dim=1, keepdim=True)) # [B, 1, 512]
                
                dynamic_rgb_prompt = self.rgb_missing_prompt + rgb_context
                dynamic_aux_prompt = self.aux_missing_prompt + aux_context
                
                xz_student[:,:self.num_patch_x,:] = xz_student[:,:self.num_patch_x,:] * rgb_mask + dynamic_rgb_prompt * (1.0 - rgb_mask)
                xz_aux_student[:,:self.num_patch_x,:] = xz_aux_student[:,:self.num_patch_x,:] * aux_mask + dynamic_aux_prompt * (1.0 - aux_mask)

                # Masking for template patches
                matrix_1 = [[1 for _ in range(5)] for _ in range(2)]
                mask = []
                for _ in range(batch_size):
                    if random.random() < (1.0 - p_miss): # template drop chance scales with p_miss as well
                        mask.append([row.copy() for row in matrix_1])
                    else:
                        mask.append(generate_random_matrix())

                xz_student[:,self.num_patch_x:,:], xz_aux_student[:,self.num_patch_x:,:] = apply_mask(
                    xz_student[:,self.num_patch_x:,:], xz_aux_student[:,self.num_patch_x:,:], mask,
                    dynamic_rgb_prompt, dynamic_aux_prompt
                )
            else:
                xz_student[:,:self.num_patch_x,:] = xz_student[:,:self.num_patch_x,:] * rgb_mask + self.rgb_missing_prompt * (1.0 - rgb_mask)
                xz_aux_student[:,:self.num_patch_x,:] = xz_aux_student[:,:self.num_patch_x,:] * aux_mask + self.aux_missing_prompt * (1.0 - aux_mask)

                # Masking for template patches
                matrix_1 = [[1 for _ in range(5)] for _ in range(2)]
                mask = []
                for _ in range(batch_size):
                    if random.random() < (1.0 - p_miss): # template drop chance scales with p_miss as well
                        mask.append([row.copy() for row in matrix_1])
                    else:
                        mask.append(generate_random_matrix())

                xz_student[:,self.num_patch_x:,:], xz_aux_student[:,self.num_patch_x:,:] = apply_mask(
                    xz_student[:,self.num_patch_x:,:], xz_aux_student[:,self.num_patch_x:,:], mask,
                    self.rgb_missing_prompt, self.aux_missing_prompt
                )

            xz_fusion_student, loss_balance_student = self.gated_fusion(torch.concat((xz_student, xz_aux_student), dim=-1), missing_mask=data)
            
            return xz_student + xz_aux_student + xz_fusion_student, loss_balance_student, xz_fusion_teacher
        else:
            # Test-time zero-channel detection and learnable prompt substitution
            rgb_is_missing = False
            aux_is_missing = False
            
            # 1. Direct label check (current search-frame status, not the
            # template history list — those are structurally different shapes
            # and would never match a [0,1]/[1,0] equality check)
            if current_missing is not None:
                if list(current_missing) == [0, 1]:
                    rgb_is_missing = True
                elif list(current_missing) == [1, 0]:
                    aux_is_missing = True
            
            # 2. Robust zero-channel detection fallback
            if not rgb_is_missing and (search_list_rgb[0].abs().mean() < 1e-5):
                rgb_is_missing = True
            if not aux_is_missing and (search_list_aux[0].abs().mean() < 1e-5):
                aux_is_missing = True

            if self.cfg.MODEL.MOE.TYPE == "CADP_GATING":
                # Compute global average pooling context from active modality
                rgb_context = self.prompt_context_mapper(xz_aux.mean(dim=1, keepdim=True)) # [B, 1, 512]
                aux_context = self.prompt_context_mapper(xz.mean(dim=1, keepdim=True)) # [B, 1, 512]
                
                dynamic_rgb_prompt = self.rgb_missing_prompt + rgb_context
                dynamic_aux_prompt = self.aux_missing_prompt + aux_context
                
                if rgb_is_missing:
                    xz[:, :self.num_patch_x, :] = dynamic_rgb_prompt
                if aux_is_missing:
                    xz_aux[:, :self.num_patch_x, :] = dynamic_aux_prompt

                missing_mask = [np.array(missing).T.tolist()]
                xz[:,self.num_patch_x:,:], xz_aux[:,self.num_patch_x:,:] = apply_mask(
                    xz[:,self.num_patch_x:,:], xz_aux[:,self.num_patch_x:,:], missing_mask,
                    dynamic_rgb_prompt, dynamic_aux_prompt
                )
            else:
                if rgb_is_missing:
                    xz[:, :self.num_patch_x, :] = self.rgb_missing_prompt
                if aux_is_missing:
                    xz_aux[:, :self.num_patch_x, :] = self.aux_missing_prompt

                missing_mask = [np.array(missing).T.tolist()]
                xz[:,self.num_patch_x:,:], xz_aux[:,self.num_patch_x:,:] = apply_mask(
                    xz[:,self.num_patch_x:,:], xz_aux[:,self.num_patch_x:,:], missing_mask,
                    self.rgb_missing_prompt, self.aux_missing_prompt
                )
            data = torch.ones(xz.shape[0], 2, device=xz.device)
            if rgb_is_missing:
                data[:, 0] = 0.0
            if aux_is_missing:
                data[:, 1] = 0.0
            xz_fusion, loss_balance = self.gated_fusion(torch.concat((xz, xz_aux), dim=-1), missing_mask=data)
            return xz + xz_aux + xz_fusion, loss_balance
    

    def forward_neck(self,enc_out,neck_h_state):
        x = enc_out
        xs = x[:, 0:self.num_patch_x]
        x,xs,h = self.neck(x,xs,neck_h_state,self.encoder.body.blocks,self.interaction_indexes)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:self.num_patch_x]
        return x,xs,h

    def forward_decoder(self, feature, loss, gt_score_map=None):
        # feature = feature[0]
        # feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   "loss":loss
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'loss': loss}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_flextrackv2(cfg):
    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    decoder = build_decoder(cfg, neck)
    model = FlexTrackV2(
        encoder,
        decoder,
        neck,
        cfg,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
    )
    # Backbone pre-train init. Only needed when TRAINING from scratch; at test
    # time the full FlexTrackV2 checkpoint is loaded afterwards and overwrites
    # these weights, so a missing pre-train file is not an error for inference.
    pretrain_path = cfg.MODEL.ENCODER.PRETRAIN_TYPE
    if pretrain_path and not os.path.exists(pretrain_path) and os.path.exists(os.getcwd() + pretrain_path):
        pretrain_path = os.getcwd() + pretrain_path
    if pretrain_path and os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    else:
        print("[build_flextrackv2] no encoder pre-train at '%s' -- skipping "
              "(fine for inference; needed only to train from scratch)." % pretrain_path)
    return model


