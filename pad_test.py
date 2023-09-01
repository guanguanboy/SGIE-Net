
import numpy as np
import torch
from torch import nn

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

def pad_test(self, window_size):        
    scale = self.opt.get('scale', 1)
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = self.lq.size()
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    self.nonpad_test(img)
    _, _, h, w = self.output.size()
    self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

def pad_size_test(window_size):
    scale = 1
    mod_pad_h, mod_pad_w = 0, 0
    input_tensor = torch.randn((1, 3, 400, 600))
    _, _, h, w = input_tensor.size()

    window_size_upsampled = window_size * 4
    if h % window_size_upsampled != 0:
        mod_pad_h = window_size_upsampled - h % window_size_upsampled
    if w % window_size_upsampled != 0:
        mod_pad_w = window_size_upsampled - w % window_size_upsampled
    img = F.pad(input_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    print('img.shape=',img.shape)
    _, _, h, w = img.size()
    output = img[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
    print('output.shape=', output.shape)

    #codes from test_sam
    input_tensor2 = torch.randn((1, 3, 400, 600))
    _, _, h, w = input_tensor2.size()
    factor = 64
    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    input_ = F.pad(input_tensor2, (0,padw,0,padh), 'reflect')
    print("paded_input shape=", input_.shape)

    restored = input_[:,:,:h,:w]
    print('restored.shape=', restored.shape)

if __name__ == '__main__':
    pad_size_test(16)