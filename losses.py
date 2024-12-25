#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:06:33 2023

@author: qifenghu
"""
import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def PDE_opt(model,x,y,nu):

    u,v,p     = model(x,y)

    # convection term
    uv_x,vu_y = torch.autograd.grad((u*v).sum(), (x,y), create_graph=True,retain_graph=True)

    uu_x  = torch.autograd.grad((u*u).sum(),x, create_graph=True,retain_graph=True)[0]
    vv_y  = torch.autograd.grad((v*v).sum(),y, create_graph=True,retain_graph=True)[0]

    # gradient of pressure
    p_x,p_y   = torch.autograd.grad(p.sum(),(x,y),create_graph=True,retain_graph=True)

    # gradient of velocity
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    v_x,v_y   = torch.autograd.grad(v.sum(),(x,y),create_graph=True,retain_graph=True)

    # Laplacian of velocity
    u_xx      = torch.autograd.grad(u_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(u_y.sum(),y,create_graph=True,retain_graph=True)[0]

    v_xx      = torch.autograd.grad(v_x.sum(),x,create_graph=True,retain_graph=True)[0]
    v_yy      = torch.autograd.grad(v_y.sum(),y,create_graph=True,retain_graph=True)[0]

    # physics loss 
    momentum_loss   = ((uu_x +  vu_y + p_x - nu * u_xx - nu * u_yy).pow(2) +\
                       (uv_x +  vv_y + p_y - nu * v_xx - nu * v_yy).pow(2)
                      ).mean(dim=0, keepdim=True)

    # incompressibility loss
    divergence_loss = (u_x + v_y).pow(2).mean(dim=0, keepdim=True)

    return momentum_loss,divergence_loss


def boundary_velocity(x,y):

    # No-slip boundary condition on all boundaries except to wall. Top walls moves at u = 1 
    u_bc = torch.zeros_like(x)
    v_bc = torch.zeros_like(x)

    idx = y==1.0
    u_bc[idx] = 1.0
    return u_bc,v_bc


def boundary_opt(model,x,y):
    u,v,_= model(x,y)
    u_bc,v_bc = boundary_velocity(x,y)
    boundary_loss = ((u - u_bc).pow(2) + (v - v_bc).pow(2)).mean(dim=0, keepdim=True)
    return boundary_loss


def pressure_opt(model,x,y):
    _,_,p= model(x,y)
    pressure_loss = p.pow(2).mean(dim=0, keepdim=True)
    return pressure_loss


