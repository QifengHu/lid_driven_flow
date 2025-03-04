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
    p_xx      = torch.autograd.grad(p_x.sum(),x,create_graph=True,retain_graph=True)[0]
    p_yy      = torch.autograd.grad(p_y.sum(),y,create_graph=True,retain_graph=True)[0]

    # gradient of velocity
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    v_x,v_y   = torch.autograd.grad(v.sum(),(x,y),create_graph=True,retain_graph=True)

    # Laplacian of velocity
    u_xx      = torch.autograd.grad(u_x.sum(),x,create_graph=True,retain_graph=True)[0]
    u_yy      = torch.autograd.grad(u_y.sum(),y,create_graph=True,retain_graph=True)[0]

    v_xx      = torch.autograd.grad(v_x.sum(),x,create_graph=True,retain_graph=True)[0]
    v_yy      = torch.autograd.grad(v_y.sum(),y,create_graph=True,retain_graph=True)[0]

    # physics loss 
    #momentum_loss   = ((uu_x +  vu_y + p_x - nu * u_xx - nu * u_yy).pow(2) +\
    #                   (uv_x +  vv_y + p_y - nu * v_xx - nu * v_yy).pow(2)
    #                  ).mean(dim=0, keepdim=True)
    mom_u_loss = (uu_x +  vu_y + p_x - nu * u_xx - nu * u_yy).pow(2)#.mean(dim=0, keepdim=True)
    mom_v_loss = (uv_x +  vv_y + p_y - nu * v_xx - nu * v_yy).pow(2)#.mean(dim=0, keepdim=True)
    # incompressibility loss
    divergence_loss = (u_x + v_y).pow(2)#.mean(dim=0, keepdim=True)
    pres_poisson_loss = (p_xx + p_yy + u_x.pow(2) + v_y.pow(2) + 2 * v_x * u_y).pow(2)
    return mom_u_loss, mom_v_loss, divergence_loss, pres_poisson_loss


def boundary_velocity(x,y):

    # No-slip boundary condition on all boundaries except to wall. Top walls moves at u = 1 
    u_bc = torch.zeros_like(x)
    v_bc = torch.zeros_like(x)

    idx = y==1.0
    u_bc[idx] = 1.0
    return u_bc,v_bc


def boundary_opt(model,x,y):
    u,v,p= model(x,y)
    
    u_bc,v_bc = boundary_velocity(x,y)
    u_boundary_loss = (u - u_bc).pow(2)#.mean(dim=0, keepdim=True)
    v_boundary_loss = (v - v_bc).pow(2)#.mean(dim=0, keepdim=True)

    # gradient of pressure
    p_x,p_y   = torch.autograd.grad(p.sum(),(x,y),create_graph=True,retain_graph=True)

    idx_x = torch.logical_or(x==0., x==1.)
    idx_y = torch.logical_or(y==0., y==1.)

    p_n_loss = torch.cat(((p_x[idx_x]-0.).pow(2),(p_y[idx_y]-0.).pow(2)),dim=0)

    return u_boundary_loss, v_boundary_loss, p_n_loss


def p_poisson_loss(model,x,y):
    u,v,p    = model(x,y)

    # gradient of velocity
    u_x,u_y   = torch.autograd.grad(u.sum(),(x,y),create_graph=True,retain_graph=True)
    v_x,v_y   = torch.autograd.grad(v.sum(),(x,y),create_graph=True,retain_graph=True)

    # gradient of pressure
    p_x,p_y   = torch.autograd.grad(p.sum(),(x,y),create_graph=True,retain_graph=True)
    p_xx      = torch.autograd.grad(p_x.sum(),x,create_graph=True,retain_graph=True)[0]
    p_yy      = torch.autograd.grad(p_y.sum(),y,create_graph=True,retain_graph=True)[0]

    pres_poisson_loss = p_xx + p_yy + u_x.pow(2) + v_y.pow(2) + 2 * v_x * u_y
    return pres_poisson_loss.pow(2)


def pressure_opt(model,x,y):
    _,_,p= model(x,y)
    pressure_loss = p.pow(2)#.mean(dim=0, keepdim=True)
    return pressure_loss


def PDE_opt2(model2,model,x,y,nu):
    u1,v1,p1     = model(x,y)
    u2,v2,p2     = model2(x,y)
    u = u1+u2
    v = v1+v2
    p = p1+p2
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
    #momentum_loss   = ((uu_x +  vu_y + p_x - nu * u_xx - nu * u_yy).pow(2) +\
    #                   (uv_x +  vv_y + p_y - nu * v_xx - nu * v_yy).pow(2)
    #                  ).mean(dim=0, keepdim=True)
    mom_u_loss = (uu_x +  vu_y + p_x - nu * u_xx - nu * u_yy).pow(2)#.mean(dim=0, keepdim=True)
    mom_v_loss = (uv_x +  vv_y + p_y - nu * v_xx - nu * v_yy).pow(2)#.mean(dim=0, keepdim=True)
    # incompressibility loss
    divergence_loss = (u_x + v_y).pow(2)#.mean(dim=0, keepdim=True)

    return mom_u_loss, mom_v_loss, divergence_loss


def boundary_opt2(model2,model,x,y):
    u1,v1,p1     = model(x,y)
    u2,v2,p2     = model2(x,y)
    u = u1+u2
    v = v1+v2
    p = p1+p2

    u_bc,v_bc = boundary_velocity(x,y)
    u_boundary_loss = (u - u_bc).pow(2)#.mean(dim=0, keepdim=True)
    v_boundary_loss = (v - v_bc).pow(2)#.mean(dim=0, keepdim=True)
    return u_boundary_loss, v_boundary_loss


def pressure_opt2(model2,model,x,y):
    u1,v1,p1     = model(x,y)
    u2,v2,p2     = model2(x,y)
    u = u1+u2 
    v = v1+v2
    p = p1+p2

    pressure_loss = p.pow(2)#.mean(dim=0, keepdim=True)
    return pressure_loss

