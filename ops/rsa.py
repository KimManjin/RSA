import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from torch.nn.init import kaiming_uniform_
from torch.autograd import Variable
from math import sqrt
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class RSA(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        nh=1,
        dk=16,
        dv=0,
        dd=0,
        kernel_size=(3,7,7),
        stride=(1,1,1),
        kernel_type='V', # ['V', 'R', 'VplusR']
        feat_type='V', # ['V', 'R', 'VplusR']
    ):
        super(RSA, self).__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        self.nh = nh
        self.dv = dv = d_out // nh if dv == 0 else dv
        self.dk = dk = dv if dk == 0 else dk
        self.dd = dd = dk if dd == 0 else dd
        
        self.kernel_size = kernel_size
        self.stride = stride   
        self.kernel_type = kernel_type
        self.feat_type = feat_type
        
        assert self.kernel_type in ['V', 'R', 'VplusR'], "Not implemented involution type: {}".format(self.kernel_type)
        assert self.feat_type in ['V', 'R', 'VplusR'], "Not implemented feature type: {}".format(self.feat_type)
        
        print("d_in: {}, d_out: {}, nh: {}, dk: {}, dv: {}, dd:{}, kernel_size: {}, kernel_type: {}, feat_type: {}"
              .format(d_in, d_out, nh, dk, dv,self.dd, kernel_size, kernel_type, feat_type))

        self.ksize = ksize = kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.pad = pad = tuple(k//2 for k in kernel_size)               
        
        # hidden dimension
        d_hid = nh * dk + dv if self.kernel_type == 'V' else nh * dk + dk + dv
        
        # Linear projection
        self.projection = nn.Conv3d(d_in, d_hid, 1, bias=False)
        
        # RSA Kernel
        if self.kernel_type == 'V':
            self.H2 = nn.Conv3d(1, dd, kernel_size, padding=self.pad, bias=False)
        elif self.kernel_type == 'R':
            self.H1 = nn.Conv3d(dk, dk*dd, kernel_size, padding=self.pad, groups=dk, bias=False)
            self.H2 = nn.Conv3d(1, dd, kernel_size, padding=self.pad, bias=False)
        elif self.kernel_type == 'VplusR':
            self.P1 = nn.Parameter(tr.randn(dk,dd).unsqueeze(0)*sqrt(1/(ksize*dd)), requires_grad=True)
            self.H1 = nn.Conv3d(dk, dk*dd, kernel_size, padding=self.pad, groups=dk, bias=False)
            self.H2 = nn.Conv3d(1, dd, kernel_size, padding=self.pad, bias=False)
        else:
            raise NotImplementedError 
        
        # Feature embedding layer
        if self.feat_type == 'V':
            pass
        elif self.feat_type == 'R':
            self.G = nn.Conv3d(1, dv, kernel_size, padding=self.pad, bias=False)
        elif self.feat_type == 'VplusR':
            self.G = nn.Conv3d(1, dv, kernel_size, padding=self.pad, bias=False)
            self.I = nn.Parameter(tr.eye(dk).unsqueeze(0), requires_grad=True)
        else:
            raise NotImplementedError 
        
        # Downsampling layer
        if max(self.stride) > 1:
            self.avgpool = nn.AvgPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))  
        
        
        
    def L2norm(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)         
    

    def forward(self, x):
        N, C, T, H, W = x.shape
        
        '''Linear projection'''
        x_proj = self.projection(x)
        
        if self.kernel_type != 'V':
            q, k, v = tr.split(x_proj, [self.nh * self.dk, self.dk, self.dv], dim=1)
        else:
            q, v = tr.split(x_proj, [self.nh * self.dk, self.dv], dim=1)
            
            
        '''Normalization'''
        q = rearrange(q, 'b (nh k) t h w -> b nh k t h w', k = self.dk) 
        q = self.L2norm(q, d=2)
        q = rearrange(q, 'b nh k t h w -> (b t h w) nh k') 
        
        v = self.L2norm(v, d=1)
                
        if self.kernel_type != 'V':
            k = self.L2norm(k, d=1)
        
        
        '''
        q = (b t h w) nh k
        k = b k t h w
        v = b v t h w
        '''
        
        # fvcore 0.1.3 only support abc,abd->acd or abc,adc->adb
            
        #RSA kernel generation
        # Basic kernel
        if self.kernel_type is 'V':
            kernel = q
        # Relational kernel
        else:
            K_H1 = self.H1(k)
            K_H1 = rearrange(K_H1, 'b (k d) t h w -> (b t h w) k d', k=self.dk)
            
            if self.kernel_type == 'VplusR':
                K_H1 = K_H1 + self.P1
            kernel = tr.einsum('abc,abd->acd', q.transpose(1,2), K_H1) # (bthw, nh, d)
        
        
        #feature generation
        # Appearance feature
        v = rearrange(v, 'b (v 1) t h w -> (b v) 1 t h w')
        
        V = self.H2(v) # (bv, d, t, h, w)
        feature = rearrange(V, '(b v) d t h w -> (b t h w) v d', v=self.dv)
        
        # Relational feature
        if self.feat_type in ['R', 'VplusR']:
            V_G = self.G(v) # (bv, v2, t, h, w)
            V_G = rearrange(V_G, '(b v) v2 t h w -> (b t h w) v v2', v=self.dv)
            
            if self.feat_type == 'VplusR':
                V_G = V_G + self.I
            
            feature = tr.einsum('abc,abd->acd', V_G, feature) # (bthw, v2, d)
        
        #kernel * feat
        out = tr.einsum('abc,adc->adb', kernel, feature) # (bthw, nh, v2)
        
        out = rearrange(out, '(b t h w) nh v -> b (nh v) t h w', t=T, h=H, w=W)
        
        if max(self.stride) > 1:
            out = self.avgpool(out)
            
        return out