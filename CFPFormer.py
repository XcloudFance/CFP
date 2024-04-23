import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from typing import Tuple, Union
from functools import partial
# device = torch.device('cuda:1')
# torch.cuda.set_device(device)

class Conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        #print("p1", x.shape)
        return self.conv(x).permute(0, 2, 3, 1)  

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x
    

class RelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        #decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        
        decay =  -1 * self.gaussian1D(self.num_heads,3)#-1*( torch.exp( -1 *(initial_value + heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads).pow(2)))

        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def gaussian1D(self,shape, sigma=1):
        m = (shape - 1.) / 2.
        x = np.linspace(-m, m, shape)
        
        h = np.exp(-(x * x) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return torch.tensor(h)


    def eu_generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2)  # (H*W, 2)

        # Eculidian Distance
        diff = grid[:, None, :] - grid[None, :, :]
        squared_diff = diff.pow(2)  # 计算差值的平方
        euclidean_distance = squared_diff.sum(dim=-1).sqrt() 
        mask = euclidean_distance * self.decay[:, None, None]  #(n, H*W, H*W)
        return mask

    def generate_1d_decay(self, l: int):
            '''
            generate 1d decay mask, the result is l*l
            '''
            index = torch.arange(l).to(self.decay)
            #print(index)
            mask = index[:, None] - index[None, :] #(l l)
            mask = mask.abs() #(l l)
            mask = mask * self.decay[:, None, None]  #(n l l)
            return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:

            retention_rel_pos = self.decay.exp()

        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = (mask_h, mask_w)

        else:
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = mask

        return retention_rel_pos
    
class GaussianAtt(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, encoder_layer, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        #mask_h, mask_w = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        #print(x.shape, v.shape, encoder_layer.shape)
      #  print('kv',k.shape, encoder_layer.shape)
        if encoder_layer != None and self.embed_dim != 64:
            k = k + encoder_layer
            v = v + encoder_layer # cross feature combination


         
        lepe = self.lepe(v) # positional encoding

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)


        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        #print(qk_mat_w)
        #print(mask_w.to(device))
        #mask_h =  torch.clamp(mask_h, min=1e-6, max=1 - 1e-7) 
        #mask_w =  torch.clamp(mask_w, min=1e-6, max=1 - 1e-7) 
        #qk_mat_w = qk_mat_w + mask_w.cuda()  #(b h n w w)
        #print(qk_mat_w)
        #qk_mat_w =  torch.clamp(qk_mat_w, min=1e-6, max=1 - 1e-7) 
        qk_mat_w = F.log_softmax(qk_mat_w, -1) #(b h n w w)
        
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        
        #qk_mat_h = qk_mat_h + mask_h.cuda()  #(b w n h h)
        #print(qk_mat_h)
        #qk_mat_h =  torch.clamp(qk_mat_h, min=1e-6, max=1 - 1e-7) 
        qk_mat_h = F.log_softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output
    

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-5,
        subln=False,
        subconv=False
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    
class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int,out_dim:int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        retention = 'chunk'
        if retention == 'chunk':
            self.retention = GaussianAtt(embed_dim, num_heads)

        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)
        #input_dim = embed_dim * 2 if embed_dim != 64 else 256
        #print(input_dim)
        #print("yeah",out_dim,embed_dim)
        self.channel_deduction = PatchEmbed(embed_dim, embed_dim)
        self.zoomin =  nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
            encoder_layer: nn.Module, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        #print('x',x.shape)
        x = x + self.pos(x)
        # if self.embed_dim == 64:

        #     encoder_layer = self.zoomin(encoder_layer)
            #print(encoder_layer.shape)
        
        encoder_layer =  self.channel_deduction(encoder_layer) if encoder_layer != None else None

        x = x + self.drop_path(self.retention(self.retention_layer_norm(x), encoder_layer, retention_rel_pos, chunkwise_recurrent, incremental_state))
        x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x
    
class UpsamplingLayer(nn.Module):

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, is_deconv = False):
        super().__init__()
        self.dim = dim
        #if is_deconv == True:
        self.reduction = nn.ConvTranspose2d(dim, out_dim, 3, 2, 1, 1) # 2 times
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x
    
class BasicLayer(nn.Module):

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 upsample: UpsamplingLayer=None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        flag = 'chunk'

        self.Relpos = RelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, out_dim, num_heads, ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if out_dim != None:
            self.upsample = upsample(dim=embed_dim , out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = upsample(dim=embed_dim , out_dim=embed_dim, norm_layer=norm_layer)

        #self.upsample = 

    def forward(self, x, encoder_layer):
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        
        for blk in self.blocks:
                x = blk(x, encoder_layer, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)

        x = self.upsample(x)
        return x
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1) #(b h w c)
        return x
    
class CFPNet(nn.Module):

    def __init__(self, in_chans=256, num_classes=1000,
                 embed_dims=[512, 256, 128, 64], out_dims = [], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, use_checkpoints=[False, False, False, False,False], chunkwise_recurrents=[True, True, True, True],
                 layerscales=[False, False, False, False,False], layer_init_values=1e-6, encoder_layers: nn.Sequential = None):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.encoder_layers = encoder_layers
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim= embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                upsample=UpsamplingLayer ,#if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoints[i_layer],
                layerscale=layerscales[i_layer - 1],
                layer_init_values=layer_init_values,
               
            )
            
            self.layers.append(layer)
        

        self.norm = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.upsample = UpsamplingLayer(2048,2048)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass


    def forward_features(self, x, encoder_layers):
        x = self.patch_embed(x)
        i = 0
        for layer in self.layers:
            #if i == 0:
            #    x = layer(x,None)
            #else:
            x = layer(x, encoder_layers[i])
            i += 1
        #x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3) #(b c h*w)
        #x = self.avgpool(x)  # B C 1
        #x = torch.flatten(x, 1)
        return x

    def forward(self, x, encoders):
        # print('more initial',x.shape)
        # x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        # x = self.upsample(x) #(b h w c)
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.forward_features(x, encoders)
        #x = self.head(x)
        return x


# @register_model
# def CrossFeaturePyramidDecoder(args):
#     model = CFPNet(
#         embed_dims=[1024, 512, 256, 64],
#         depths=[2, 2, 8, 2],
#         num_heads=[4, 4, 8, 16],
#         init_values=[2, 2, 2, 2],
#         heads_ranges=[4, 4, 6, 6],
#         mlp_ratios=[3, 3, 3, 3],
#         drop_path_rate=0.1,
#         chunkwise_recurrents=[True, True, True, True],
#         layerscales=[False, False, False, False]
#     )
#     model.default_cfg = _cfg()
#     return model


# @register_model
# def RMT_S(args):
#     model = VisRetNet(
#         embed_dims=[64, 128, 256, 512],
#         depths=[3, 4, 18, 4],
#         num_heads=[4, 4, 8, 16],
#         init_values=[2, 2, 2, 2],
#         heads_ranges=[4, 4, 6, 6],
#         mlp_ratios=[4, 4, 3, 3],
#         drop_path_rate=0.15,
#         chunkwise_recurrents=[True, True, True, False],
#         layerscales=[False, False, False, False]
#     )
#     model.default_cfg = _cfg()
#     return model


# @register_model
# def RMT_B(args):
#     model = VisRetNet(
#         embed_dims=[80, 160, 320, 512],
#         depths=[4, 8, 25, 8],
#         num_heads=[5, 5, 10, 16],
#         init_values=[2, 2, 2, 2],
#         heads_ranges=[5, 5, 6, 6],
#         mlp_ratios=[4, 4, 3, 3],
#         drop_path_rate=0.4,
#         chunkwise_recurrents=[True, True, True, False],
#         layerscales=[False, False, True, True],
#         layer_init_values=1e-6
#     )
#     model.default_cfg = _cfg()
#     return model


# @register_model
# def RMT_L(args):
#     model = VisRetNet(
#         embed_dims=[112, 224, 448, 640],
#         depths=[4, 8, 25, 8],
#         num_heads=[7, 7, 14, 20],
#         init_values=[2, 2, 2, 2],
#         heads_ranges=[6, 6, 6, 6],
#         mlp_ratios=[4, 4, 3, 3],
#         drop_path_rate=0.5,
#         chunkwise_recurrents=[True, True, True, False],
#         layerscales=[False, False, True, True],
#         layer_init_values=1e-6
#     )
#     model.default_cfg = _cfg()
#     return model
