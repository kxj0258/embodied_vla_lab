#!/usr/bin/env python3
"""Small convolutional latent world-model components for Phase 5."""
import torch
from torch import nn

class ImageEncoder(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__(); self.conv=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.SiLU(),nn.Conv2d(16,32,4,2,1),nn.SiLU(),nn.Conv2d(32,48,4,2,1),nn.SiLU(),nn.Conv2d(48,64,4,2,1),nn.SiLU()); self.proj=nn.Linear(64*8*8,latent_dim)
    def forward(self,x): return self.proj(self.conv(x).flatten(1))

class ImageDecoder(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__(); self.proj=nn.Linear(latent_dim,64*8*8); self.deconv=nn.Sequential(nn.ConvTranspose2d(64,48,4,2,1),nn.SiLU(),nn.ConvTranspose2d(48,32,4,2,1),nn.SiLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.SiLU(),nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid())
    def forward(self,z): return self.deconv(self.proj(z).reshape(-1,64,8,8))

class ActionFreeDynamics(nn.Module):
    def __init__(self,latent_dim=128,hidden_dim=256):
        super().__init__(); self.delta=nn.Sequential(nn.Linear(latent_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,latent_dim))
    def step(self,z,action=None): return z+self.delta(z)
    def rollout(self,z,actions):
        out=[]
        for t in range(actions.shape[1]): z=self.step(z); out.append(z)
        return torch.stack(out,1)

class ActionConditionedDynamics(nn.Module):
    def __init__(self,latent_dim=128,action_dim=7,hidden_dim=256):
        super().__init__(); self.action=nn.Sequential(nn.Linear(action_dim,32),nn.SiLU()); self.delta=nn.Sequential(nn.Linear(latent_dim+32,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,latent_dim))
    def step(self,z,action): return z+self.delta(torch.cat([z,self.action(action)],-1))
    def rollout(self,z,actions):
        out=[]
        for t in range(actions.shape[1]): z=self.step(z,actions[:,t]); out.append(z)
        return torch.stack(out,1)

class SmallWorldModel(nn.Module):
    def __init__(self,conditioned=True,latent_dim=128,action_dim=7,hidden_dim=256):
        super().__init__(); self.encoder=ImageEncoder(latent_dim); self.decoder=ImageDecoder(latent_dim); self.dynamics=ActionConditionedDynamics(latent_dim,action_dim,hidden_dim) if conditioned else ActionFreeDynamics(latent_dim,hidden_dim)
    def reconstruct(self,image): return self.decoder(self.encoder(image))
    def rollout(self,image,actions):
        z=self.encoder(image); zs=self.dynamics.rollout(z,actions); b,h,d=zs.shape; return self.decoder(zs.reshape(b*h,d)).reshape(b,h,3,128,128)

def count_parameters(module): return sum(p.numel() for p in module.parameters())
