# -*- coding: utf-8 -*-

""" 
Volume Segmentation

Team Challenge Group 2
R.E. Buijs, D.M Cornelissen, D. Devetzis, D. Le, J. Zhang
Utrecht University & University of Technology Eindhoven

""" 

import torch
import torchvision
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p 

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        if x.shape != skip.shape:
            x = torchvision.transforms.functional.resize(x, size=skip.shape[2:])
#         print(x.shape, skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Downsampling """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        self.b = conv_block(512, 1024)

        """ Upsampling """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Output """
        self.output_conv = nn.Conv2d(64, 25, kernel_size=1, padding=0) 
#         self.output_softmax = nn.Softmax(dim=1) No need because loss function expects raw logits

    def forward(self, inputs):
        """ Downsampling """
        s1, p1 = self.e1(inputs)
        print(f's1: {s1.shape}, p1: {p1.shape}')
        s2, p2 = self.e2(p1)
        print(f's2: {s2.shape}, p2: {p2.shape}')
        s3, p3 = self.e3(p2)
        print(f's3: {s3.shape}, p3: {p3.shape}')
        s4, p4 = self.e4(p3)
        print(f's4: {s4.shape}, p4: {p4.shape}')

        b = self.b(p4)
        print(f'b: {b.shape}, p4: {p4.shape}')

        """ Upsampling """
        d1 = self.d1(b, s4)
        print(f'b: {b.shape},s4: {s4.shape}')
        d2 = self.d2(d1, s3)
#         print(f'b: {d1.shape},s4: {s3.shape}')
        d3 = self.d3(d2, s2)
#         print(f'b: {d2.shape},s4: {s2.shape}')
        d4 = self.d4(d3, s1)
#         print(f'b: {d3.shape},s4: {s1.shape}')

        """ Output """
        outputs = self.output_conv(d4)
#         outputs = self.output_softmax(outputs) No need

        return outputs