import torch
from torch import nn
import random, numpy as np

class ConvEmbed(nn.Module):
    def __init__(self, in_ch, dim, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, dim, kernel, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x.transpose(1,2).reshape(B,C,H,W)


class ConvProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1,groups=dim),
            nn.BatchNorm2d(dim)
        )

    def forward(self,x,H,W):
        B,N,C=x.shape
        x=x.transpose(1,2).reshape(B,C,H,W)
        x=self.conv(x)
        return x.flatten(2).transpose(1,2)


class ConvAttention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.heads=heads
        self.scale=(dim//heads)**-0.5

        self.q_conv=ConvProjection(dim)
        self.k_conv=ConvProjection(dim)
        self.v_conv=ConvProjection(dim)

        self.q=nn.Linear(dim,dim)
        self.k=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim)
        self.proj=nn.Linear(dim,dim)

    def forward(self,x,H,W):
        B,N,C=x.shape
        h=self.heads
        d=C//h

        q=self.q(self.q_conv(x,H,W))
        k=self.k(self.k_conv(x,H,W))
        v=self.v(self.v_conv(x,H,W))

        q=q.reshape(B,N,h,d).transpose(1,2)
        k=k.reshape(B,N,h,d).transpose(1,2)
        v=v.reshape(B,N,h,d).transpose(1,2)

        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(-1)

        x=attn@v
        x=x.transpose(1,2).reshape(B,N,C)

        return self.proj(x)


class CvTBlock(nn.Module):
    def __init__(self,dim,heads,mlp_ratio=4,drop=.1):
        super().__init__()

        self.norm1=nn.LayerNorm(dim)
        self.attn=ConvAttention(dim,heads)

        self.norm2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(
            nn.Linear(dim,dim*mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim*mlp_ratio,dim),
            nn.Dropout(drop)
        )

    def forward(self,x,H,W):
        x=x+self.attn(self.norm1(x),H,W)
        x=x+self.mlp(self.norm2(x))
        return x


class CvTStage(nn.Module):
    def __init__(self,in_ch,dim,kernel,stride,padding,heads,depth):
        super().__init__()

        self.embed=ConvEmbed(in_ch,dim,kernel,stride,padding)

        self.blocks=nn.ModuleList([
            CvTBlock(dim,heads)
            for _ in range(depth)
        ])

    def forward(self,x):
        x=self.embed(x)

        B,C,H,W=x.shape
        x=x.flatten(2).transpose(1,2)

        for block in self.blocks:
            x=block(x,H,W)

        return x.transpose(1,2).reshape(B,C,H,W)


class CvT(nn.Module):
    def __init__(self,classes=10):
        super().__init__()

        self.stage1=CvTStage(
            3,64,
            kernel=7,
            stride=4,
            padding=2,
            heads=1,
            depth=1
        )

        self.stage2=CvTStage(
            64,192,
            kernel=3,
            stride=2,
            padding=1,
            heads=3,
            depth=2
        )

        self.stage3=CvTStage(
            192,384,
            kernel=3,
            stride=2,
            padding=1,
            heads=6,
            depth=4
        )

        self.norm=nn.LayerNorm(384)
        self.fc=nn.Linear(384,classes)

    def forward(self,x):
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)

        x=x.flatten(2).transpose(1,2)
        x=self.norm(x)
        x=x.mean(1)

        return self.fc(x)