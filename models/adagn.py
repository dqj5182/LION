import torch.nn as nn
from utils.checker import *
from .dense import dense

class AdaGN(nn.Module):
    '''
    adaptive group normalization
    '''
    def __init__(self, ndim, cfg, n_channel):
        """
        ndim: dim of the input features 
        n_channel: number of channels of the inputs 
        ndim_style: channel of the style features 
        """
        super().__init__()
        style_dim = cfg.latent_pts.style_dim 
        init_scale = cfg.latent_pts.ada_mlp_init_scale 
        self.ndim = ndim 
        self.n_channel = n_channel
        self.style_dim = style_dim
        self.out_dim = n_channel * 2
        self.norm = nn.GroupNorm(8, n_channel)
        in_channel = n_channel 
        self.emd = dense(style_dim, n_channel*2, init_scale=init_scale)
        self.emd.bias.data[:in_channel] = 1
        self.emd.bias.data[in_channel:] = 0

    def __repr__(self):
        return f"AdaGN(GN(8, {self.n_channel}), Linear({self.style_dim}, {self.out_dim}))" 
        
    def forward(self, image, style):
        # style: B,D 
        # image: B,D,N,1 
        CHECK2D(style)
        style = self.emd(style)
        if self.ndim == 3: #B,D,V,V,V
            CHECK5D(image)
            style = style.view(style.shape[0], -1, 1, 1, 1) # 5D 
        elif self.ndim == 2: # B,D,N,1 
            CHECK4D(image) 
            style = style.view(style.shape[0], -1, 1, 1) # 4D 
        elif self.ndim == 1: # B,D,N
            CHECK3D(image) 
            style = style.view(style.shape[0], -1, 1) # 4D 
        else:
            raise NotImplementedError

        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias  
        return result 


