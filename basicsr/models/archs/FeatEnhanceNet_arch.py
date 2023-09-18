# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from Fuse_Block import FeatFuseBlock

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.fuse_transform = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

            self.fuse.append(FeatFuseBlock(chan,chan))
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )

            self.fuse_transform.append(nn.Conv2d(chan, 2*chan, 2, 2))

            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        # three conv fusion layers for obtaining HQ-Feature
        vit_dim = 1280
        transformer_dim = 256
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )        

    def forward(self, inp, inter_feats, deep_feats):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        ealsy_inter_feat = inter_feats[0]
        hq_features_list = []
        
        vit_features = ealsy_inter_feat.permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(deep_feats) + self.compress_vit_feat(vit_features)
        #hq_features_list.append(hq_features)
        
        encs = []

        for encoder, fuse, down, fuse_transform in zip(self.encoders, self.fuse, self.downs, self.fuse_transform):
            fused_x = fuse(x, hq_features)
            x = encoder(fused_x)
            encs.append(x)
            x = down(x)
            hq_features = fuse_transform(hq_features)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp[:,:3,:,:]

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
class FeatEnhanceNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        sam_checkpoint = "./segment_anything/pretrained_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.freezed_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        #self.mask_generator = SamAutomaticMaskGenerator(self.freezed_sam)

        # freeze pretrained model
        for param in self.freezed_sam.parameters():
            param.requires_grad = False

        self.naf_enhancenet = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)

        #image encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width//2, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.conv1_1 = nn.Conv2d(in_channels=width//2, out_channels=width//2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
    def forward(self, input): #这里mask直接使用字典吧

        #torch.autograd.set_detect_anomaly(True)

        image = input[:,0:3,:,:].clone()
        masks = input[:,3:,:,:].clone().bool()

        #get intermediate features
        batched_input = []
        for b_i in range(len(image)):
            dict_input = dict()
            input_image = torch.as_tensor((image[b_i]*255).to(torch.uint8), device=self.freezed_sam.device).contiguous() #我们只需要保证输入的是0到255的整型troch即可，归一化会在sam self.preprocess里面做。
            dict_input['image'] = input_image 
            dict_input['original_size'] = image[b_i].shape[1:3]
            batched_input.append(dict_input)

        batched_output, interm_embeddings, image_embeddings = self.freezed_sam(batched_input, multimask_output=False)

        x = self.naf_enhancenet(image, interm_embeddings, image_embeddings)

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
if __name__ == '__main__':
    img_channel = 19
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    #net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      #enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()


    #inp_shape = (3, 256, 256)

    inp_img = torch.randn(1, 3, 256, 256).cuda()
    #output = net(inp_img)
    #print(output.shape)

    samenhancenet = FeatEnhanceNet(img_channel=3, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()
    
    input_sam_image = torch.randn(2, 3, 256, 256).cuda()

    inter_meditate_feats = torch.randn(2, 11, 256, 256).cuda()
    output_sam = samenhancenet(input_sam_image)
    print(output_sam.shape)
