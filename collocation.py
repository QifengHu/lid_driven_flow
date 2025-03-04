#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:01:14 2023

@author: qifenghu
"""
import torch
import numpy as np
#from scipy.interpolate import interp1d
from losses import PDE_opt
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
dtype  = torch.float64


def sample_collocation_points(domain, n_dom):
    dim      = domain.shape[0]
    soboleng = torch.quasirandom.SobolEngine(dimension=dim,scramble=True)
    data     = soboleng.draw(n_dom, dtype=torch.float64)*(domain[1] - domain[0]) + domain[0]
    x        = data[:,0][:,None]
    y        = data[:,1][:,None]

    #x_bc, y_bc = sample_boundary_points(domain, n_bc)
    #x = torch.cat((x, x_bc), dim=0)
    #y = torch.cat((y, y_bc), dim=0)
    return x,y


def sample_boundary_points(domain,batch_size):
    x_min =domain[0][0]
    x_max =domain[1][0]

    y_min =domain[0][1]
    y_max =domain[1][1]

    soboleng = torch.quasirandom.SobolEngine(dimension=1,scramble=True)

    x_top   = soboleng.draw(batch_size,dtype=torch.float64)*(x_max - x_min) + x_min
    y_top   = torch.full_like(x_top,y_max)

    y_right = soboleng.draw(batch_size,dtype=torch.float64)*(y_max - y_min) + y_min
    x_right = torch.full_like(y_right,x_max)

    x_bottom = soboleng.draw(batch_size,dtype=torch.float64)*(x_max - x_min) + x_min
    y_bottom = torch.full_like(x_bottom,y_min)

    y_left = soboleng.draw(batch_size,dtype=torch.float64)*(y_max - y_min) + y_min
    x_left = torch.full_like(y_left,x_min)

    x = torch.cat((x_top,x_right,x_bottom,x_left),dim=0)
    y = torch.cat((y_top,y_right,y_bottom,y_left),dim=0)

    # include corner points
    #y[-1] = y_max
    #x[-1] = x_min
    #y[-2] = y_max
    #x[-2] = x_max

    #y[-3] = y_min
    #x[-3] = x_min
    #y[-4] = y_min
    #x[-4] = x_max
    
    return x,y


def adaptive_collocation(domain, n_dom, model, nu, n_incre, far_neighbor=False):
    x,y  = sample_collocation_points(domain, n_dom)
    x       = x.to(device).requires_grad_(True)
    y       = y.to(device).requires_grad_(True)

    _, _, divergence_loss = PDE_opt(model,x,y,nu)
    
    descending_idx  = torch.argsort(divergence_loss, dim=0, descending=True)
    half_descending_idx = descending_idx[:n_dom//2]
    if far_neighbor == False:
        top_descending_idx = half_descending_idx[:n_incre]
        x_extra = x[top_descending_idx].reshape(-1,1).detach()
        y_extra = y[top_descending_idx].reshape(-1,1).detach()
    else:
        x_half = x[half_descending_idx].detach().flatten()
        y_half = y[half_descending_idx].detach().flatten()
        new_x  = [x_half[0]]
        new_y  = [y_half[0]]
        previous_idx = 0
        
        for _ in range(n_incre-1):
            mask = torch.ones(len(x_half), dtype=torch.bool)

            distances = torch.sqrt((x_half - new_x[-1]).pow(2)+(y_half - new_y[-1]).pow(2))
            furthest_neighbor_idx = torch.argmax(distances)            
            new_x.append( x_half[furthest_neighbor_idx] )
            new_y.append( y_half[furthest_neighbor_idx] )

            mask[previous_idx] = False
            x_half = x_half[mask]
            y_half = y_half[mask]
            previous_idx = furthest_neighbor_idx

        # Convert the list of new points into PyTorch tensors
        x_extra = torch.stack(new_x).unsqueeze(1)
        y_extra = torch.stack(new_y).unsqueeze(1)
    return x_extra, y_extra



'''

##### For interior point distribution [64,64]: space: 66 points per side
def sample_collocation_points(domain, dom_dis):
    x_min    = domain[0][0]
    x_max    = domain[1][0]
    y_min    = domain[0][1]
    y_max    = domain[1][1]

    x_unique = torch.linspace( x_min, x_max, dom_dis[0]+2 )[1:-1]
    y_unique = torch.linspace( y_min, y_max, dom_dis[1]+2 )[1:-1]
    
    y,x      = torch.meshgrid(y_unique, x_unique, indexing='ij')
    x        = x.reshape(-1,1)
    y        = y.reshape(-1,1)
    return x,y
    
##### boundary point distribution [128, 4]: 130 points per space side
def sample_boundary_points(domain, bc_dis):
    x_min    = domain[0][0]
    x_max    = domain[1][0]
    y_min    = domain[0][1]
    y_max    = domain[1][1]

    x_unique = torch.linspace( x_min, x_max, bc_dis[0]+2 )[1:-1]
    y_unique = torch.linspace( y_min, y_max, bc_dis[0]+2 )[1:-1]

    # x_min & x_max BC
    y_x      = y_unique.reshape(-1,1)
    x_y_min  = torch.full_like(y_x, x_min)
    x_y_max  = torch.full_like(y_x, x_max)
    
    # y_min & y_max BC
    x_y      = x_unique.reshape(-1,1)
    y_x_min  = torch.full_like(x_y, y_min)
    y_x_max  = torch.full_like(x_y, y_max)
    
    x      = torch.cat((x_y_min, x_y_max, x_y,     x_y    ), dim=0)
    y      = torch.cat((y_x,     y_x,     y_x_min, y_x_max), dim=0)
    return x,y


'''  
