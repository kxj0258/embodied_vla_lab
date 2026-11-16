#!/usr/bin/env python3
"""Copy-preserving residual latent world models for Phase 5D."""
import torch
from torch import nn
from phase5_small_world_model import ImageEncoder,ActionFreeDynamics,ActionConditionedDynamics,count_parameters

class ResidualDecoder(nn.Module):
    def __init__(self,latent_dim=128,max_abs_residual=.25):
        super().__init__(); self.max_abs_residual=max_abs_residual; self.proj=nn.Linear(latent_dim,64*8*8); self.body=nn.Sequential(nn.ConvTranspose2d(64,48,4,2,1),nn.SiLU(),nn.ConvTranspose2d(48,32,4,2,1),nn.SiLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.SiLU()); self.head=nn.ConvTranspose2d(16,3,4,2,1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,z):
        x=self.proj(z).reshape(-1,64,8,8); return torch.tanh(self.head(self.body(x)))*self.max_abs_residual

class ResidualWorldModel(nn.Module):
    def __init__(self,conditioned=True,latent_dim=128,action_dim=7,hidden_dim=256,max_abs_residual=.25):
        super().__init__(); self.encoder=ImageEncoder(latent_dim); self.decoder=ResidualDecoder(latent_dim,max_abs_residual); self.dynamics=ActionConditionedDynamics(latent_dim,action_dim,hidden_dim) if conditioned else ActionFreeDynamics(latent_dim,hidden_dim)
    def latent_rollout(self,image,actions): return self.dynamics.rollout(self.encoder(image),actions)
    def rollout(self,image,actions):
        zs=self.latent_rollout(image,actions); b,h,d=zs.shape; delta=self.decoder(zs.reshape(b*h,d)).reshape(b,h,3,128,128); return torch.clamp(image[:,None]+delta,0,1)

__all__=["ResidualWorldModel","count_parameters"]
