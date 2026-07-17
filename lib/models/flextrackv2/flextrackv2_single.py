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

import torch

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
    def __init__(self, embed_size, hidden_size):
        super(LinearAttention_moe, self).__init__()
        embed_size = embed_size*2
        hidden_size = hidden_size*2
        self.query_proj = nn.Linear(embed_size, hidden_size)
        self.key_proj = nn.Linear(embed_size, hidden_size)
        self.value_proj = MoEFusion()
        self.g_proj = nn.Linear(embed_size, hidden_size)

        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, int(embed_size/2))

        

    def forward(self, x):
        # Linear projections
        Q = self.silu(self.query_proj(x))  # Query
        K = self.silu(self.key_proj(x))    # Key



        # V = self.silu(self.value_proj(x))  # Value
        V, balance_loss = self.value_proj(x)
        V = self.silu(V)

        
        G = F.softmax(self.g_proj(x),dim=-1)  # Gate (softmax activation)



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
        G = F.softmax(self.g_proj(x),dim=-1)  # Gate (softmax activation)



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
        # self.moefusion = MoEFusion()
        self.gated_fusion = LinearAttention_moe(512,512)



    def forward(self, template_list=None, search_list=None, template_anno_list=None,enc_opt=None,neck_h_state=None, feature=None, mode="encoder",loss=None):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            xz, loss = self.forward_encoder(template_list, search_list, template_anno_list)
            self.loss = loss
            return xz,loss
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state)
        elif mode == "decoder":
            

            return self.forward_decoder(feature,self.loss)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, template_anno_list):
        # Forward the encoder
        template_list_rgb = [tensor[:,:3, :, :] for tensor in template_list]
        template_list_aux = [tensor[:,3:, :, :] for tensor in template_list]
        search_list_rgb = [tensor[:,:3, :, :] for tensor in search_list]
        search_list_aux = [tensor[:,3:, :, :] for tensor in search_list]
        
        

        
        xz = self.encoder(template_list_rgb, search_list_rgb, template_anno_list)

        xz_aux = self.encoder(template_list_aux, search_list_aux, template_anno_list)


 


        if self.training:
            random_number = random.choice([1, 2, 3])
            possible_combinations = torch.tensor([
                                            [0, 1],
                                            [1, 0],
                                            [1, 1],
                                            [1, 1],
                                            [1, 1],
                                            [1, 1]
                                        ])

            # random mask rgb and x search patch
            batch_size = xz.shape[0]
            indices = torch.randint(0, 6, (batch_size,))  
            data = possible_combinations[indices] 

            xz[:,:self.num_patch_x,:] = xz[:,:self.num_patch_x,:]*data[:,0].unsqueeze(-1).unsqueeze(-1).expand_as(xz[:,:self.num_patch_x,:]).to(xz.device)
            xz_aux[:,:self.num_patch_x,:] = xz_aux[:,:self.num_patch_x,:]*data[:,1].unsqueeze(-1).unsqueeze(-1).expand_as(xz[:,:self.num_patch_x,:]).to(xz.device)
            

            xz[:,self.num_patch_x:,:], mask1 = random_mask(xz[:,self.num_patch_x:,:], mask_prob=0.5, num_tokens_to_mask=122)
            xz_aux[:,self.num_patch_x:,:], mask2 = random_mask(xz_aux[:,self.num_patch_x:,:], mask_prob=0.5, num_tokens_to_mask=122)




        xz_fusion, loss_balance = self.gated_fusion(torch.concat((xz,xz_aux),dim=-1))

        # xz_fusion,loss_balance = self.moefusion(xz_fusion)
        






        return xz+xz_aux+xz_fusion, loss_balance
    


    def forward_neck(self,enc_out,neck_h_state):
        x = enc_out
        xs = x[:, 0:self.num_patch_x]
        x,xs,h = self.neck(x,xs,neck_h_state,self.encoder.body.blocks,self.interaction_indexes)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:self.num_patch_x]
        return x,xs,h

    def forward_decoder(self, feature, loss,gt_score_map=None):
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
    checkpoint = torch.load(cfg.MODEL.ENCODER.PRETRAIN_TYPE, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    return model


