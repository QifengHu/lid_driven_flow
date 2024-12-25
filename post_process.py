import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata

# https://joseph-long.com/writing/colorbars/
def colorbar(mappable,min_val,max_val,limit):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    ticks = np.linspace(min_val, max_val, 4, endpoint=True)
    cbar = fig.colorbar(mappable, cax=cax,ticks=ticks)
    cbar.formatter.set_powerlimits((limit, limit))
    plt.sca(last_axes)
    return cbar

params = {
    'text.latex.preamble': '\\usepackage{gensymb}',
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels
    'axes.titlesize': 14,
    'font.size'     : 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': False,
    'figure.figsize': [4, 4],
    'font.family': 'serif',
}

cmap_list = ['jet','YlGnBu','coolwarm','rainbow','magma','plasma','inferno','Spectral','RdBu']


# configurations
plt.rcParams.update(params)
cmap = plt.cm.get_cmap(cmap_list[8]).reversed()


# In[28]:

def contour_vel_mag(model, device, trial, methodname):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(1, 1)
    
    ax = plt.subplot(gs[0,0])
    
    x = torch.linspace(0,1,128)
    y = torch.linspace(0,1,128)
    
    x,y = torch.meshgrid(x,y,indexing='xy')
    
    up,vp,_ = model(x.reshape(-1,1).to(device),y.reshape(-1,1).to(device))
    
    up = up.reshape(x.shape).cpu()
    vp = vp.reshape(x.shape).cpu()
    
    velocity = (up.detach().pow(2) + vp.detach().pow(2)).sqrt()
    
    max_val = 1.0 #torch.amax(velocity)
    min_val = 0.0 #torch.amin(velocity)
    
    pimg=plt.pcolormesh(x.numpy(),y.numpy(),
                        velocity,cmap=cmap,
                        shading='gouraud',
                        vmin=min_val,
                        vmax=max_val)
    
    colorbar(pimg,min_val = min_val,max_val= max_val,limit=-1)
    ax.axis('scaled')
    
    x = torch.linspace(0,1,256)
    y = torch.linspace(0,1,256)
    
    x,y = torch.meshgrid(x,y,indexing='xy')
    up,vp,pp = model(x.reshape(-1,1).to(device),y.reshape(-1,1).to(device))
    
    up = up.reshape(x.shape).cpu()
    vp = vp.reshape(x.shape).cpu()
    
    plt.streamplot(x.numpy(),y.numpy(),
                      up.detach().numpy(),
                      vp.detach().numpy(),
                      density=2.,
                      arrowsize=0.5,
                      arrowstyle='->',
                      color='w',
                      linewidth=0.5,
                      cmap=cmap)
    ax.axis('scaled')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xticks(np.linspace(0,1,3))
    ax.set_yticks(np.linspace(0,1,3))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_contour_vel_mag.png',bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_horiz_vel(model, device, trial, methodname):
    fig = plt.figure(figsize=(3,3))
    gs = gridspec.GridSpec(1, 1)
    
    ax = plt.subplot(gs[0,0])
    
    y = []
    u = []
    with open('benchmarks/ghiau.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            y.append(float(line.split()[0]))
            u.append(float(line.split()[1]))
    
    y = torch.tensor(y).reshape(-1,1)
    x = torch.full_like(y,0.5)
    u = torch.tensor(u).reshape(-1,1)
    
    # prediction from the model
    y_ = torch.linspace(0,1,256)[:,None]
    x_ = torch.full_like(y_,0.5)
    up,_,_= model(x_.to(device),y_.to(device))
    
    
    ax.plot(up.cpu().detach().numpy(),y_,"m",linewidth=3,label='Predicted')
    ax.plot(u,y,'xk',label="Ghia et al.",markersize=6)
    ax.set_xlabel('$u(0.5,y)$')
    ax.set_ylabel('$y$')
    ax.set_yticks([0,0.5,1])
    ax.set_xticks([-0.5,0.5,0.5,1])
    ax.legend(frameon=False,fontsize=12)
    plt.savefig(f'pic/{methodname}_{trial}_plot_verif_horiz_vel.png',bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_vert_vel(model, device, trial, methodname):
    fig = plt.figure(figsize=(3,3))
    gs = gridspec.GridSpec(1, 1)
    
    ax = plt.subplot(gs[0,0])
    
    # model
    x = torch.linspace(0,1,256)[:,None]
    y = torch.full_like(x,0.5)
    _,vp,_= model(x.to(device),y.to(device))
    
    ax.plot(x,vp.cpu().detach(),"m",linewidth=3,label="Predicted")
    
    
    x = []
    v = []
    with open('benchmarks/ghiav.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            x.append(float(line.split()[0]))
            v.append(float(line.split()[1]))
    
    x = torch.tensor(x)
    v = torch.tensor(v)
    ax.plot(x,v,'xk',label='Ghia et al.',markersize=6)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v(x,0.5)$')
    ax.set_yticks([-0.3,0,0.3])
    ax.set_xticks([-0,0.5,1.0])
    ax.legend(frameon=False,fontsize=12)
    plt.savefig(f'pic/{methodname}_{trial}_plot_verif_vert_vel.png',bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_update(trial, methodname):
    obj_s    = np.loadtxt(f'data/{trial}_object.dat')
    mu_s     = np.loadtxt(f'data/{trial}_mu.dat')
    constr_s = np.loadtxt(f'data/{trial}_constr.dat')
    lambda_s = np.loadtxt(f'data/{trial}_lambda.dat')

    linestyles = ['-', ':', '-.', '--']
    colors = ['k', 'b', 'g', 'r']  
    markers = ['o', 's', '^', 'd'] 

    num_all   = obj_s.shape[0]
    #Plot loss terms evolution
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(obj_s[:num_all,1], label=r'$\mathcal{J}$', linestyle=linestyles[0], color=colors[0],
            linewidth=2, marker=markers[0], markersize=5, markevery=num_all//10)
    ax.plot(constr_s[:num_all,1], label=r'$\mathcal{C}_{P}$', linestyle=linestyles[1], color=colors[1],
            linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    ax.plot(constr_s[:num_all,2], label=r'$\mathcal{C}_{B}$', linestyle=linestyles[2], color=colors[2],
            linewidth=2, marker=markers[2], markersize=5, markevery=num_all//10)
    ax.plot(constr_s[:num_all,3], label=r'$\mathcal{C}_{M}$', linestyle=linestyles[3], color=colors[3],
            linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Losses', color='k')
    ax.semilogy()
    ax.set_ylim(1e-8, 1e0)
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    #plt.grid()
    plt.savefig(f'pic/{methodname}_{trial}_losses.png', dpi=300)
    plt.close()

    #Plot lambda & mu evolution
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(lambda_s[:num_all,1], label=r'$\lambda_{P}$', linestyle=linestyles[1], color=colors[1],
            linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    ax.plot(lambda_s[:num_all,2], label=r'$\lambda_{B}$', linestyle=linestyles[2], color=colors[2],
            linewidth=2, marker=markers[2], markersize=5, markevery=num_all//10)
    ax.plot(lambda_s[:num_all,3], label=r'$\lambda_{M}$', linestyle=linestyles[3], color=colors[3],
            linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Lagrange Mult.', color='k')
    ax.semilogy()
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    #plt.grid()
    plt.savefig(f'pic/{methodname}_{trial}_lambda.png', dpi=300)
    plt.close()

    #Plot mu evolution
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(mu_s[:num_all,1], label=r'$\mu_{P}$', linestyle=linestyles[1], color=colors[1],
            linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    ax.plot(mu_s[:num_all,2], label=r'$\mu_{B}$', linestyle=linestyles[2], color=colors[2],
            linewidth=2, marker=markers[2], markersize=5, markevery=num_all//10)
    ax.plot(mu_s[:num_all,3], label=r'$\mu_{M}$', linestyle=linestyles[3], color=colors[3],
            linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Penalty Para.', color='k')
    ax.semilogy()
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    #plt.grid()
    plt.savefig(f'pic/{methodname}_{trial}_mu.png', dpi=300)
    plt.close()

