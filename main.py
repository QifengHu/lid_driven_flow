#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from tqdm import tqdm
torch.manual_seed(0)
from losses import PDE_opt, boundary_opt, pressure_opt
from collocation import sample_collocation_points, sample_boundary_points
from post_process import contour_vel_mag, plot_horiz_vel, plot_vert_vel, plot_update

# multiple random restarts to improve optimization quality with multi-grid type data


# In[6]:


class non_linear_layer(torch.nn.Module):
    def __init__(self,in_N: int,out_N: int):
        super(non_linear_layer, self).__init__()
        self.Ls  = None
        self.net =torch.nn.Sequential(torch.nn.Linear(in_N,out_N),torch.nn.Tanh()) 

    def forward(self, x):
        out = self.net(x)
        return out 

class navier_stokes_architecture(torch.nn.Module):
    def __init__(self,layers,**kwargs):
        super(navier_stokes_architecture,self).__init__()

        self.mean  = torch.nn.Parameter(kwargs["mean"],requires_grad=False)
        self.stdev = torch.nn.Parameter(kwargs["stdev"],requires_grad=False)
        
        _layers_u = [] 
        _layers_v = []
        _layers_p = []
        
        for i in range(0, len(layers) - 2):
            _layers_u.append(non_linear_layer(layers[i], layers[i + 1]))
            _layers_v.append(non_linear_layer(layers[i], layers[i + 1]))
            _layers_p.append(non_linear_layer(layers[i], layers[i + 1]))
        
        _layers_u.append(torch.nn.Linear(layers[-2], layers[-1]))
        _layers_v.append(torch.nn.Linear(layers[-2], layers[-1]))
        _layers_p.append(torch.nn.Linear(layers[-2], layers[-1]))
        
        self.net_u = torch.nn.Sequential(*_layers_u)
        self.net_v = torch.nn.Sequential(*_layers_v)
        self.net_p = torch.nn.Sequential(*_layers_p)
        
    def forward(self,x,y):
        data = torch.cat((x,y),dim=1);
        # normalize the input
        data = (data - self.mean)/self.stdev
        out  = self.net_u(data), self.net_v(data), self.net_p(data)
        return out

def Xavier_initialization(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)  

def stats(domain_coords):
    dim      = len(domain_coords)
    coords_mean = (domain_coords[1,:] + domain_coords[0,:]) / 2
    coords_std  = (domain_coords[1,:] - domain_coords[0,:]) / np.sqrt(12)
    return np.vstack((coords_mean, coords_std))


# In[15]:


def modelling(domain, layers, device):
    coords_stat = stats(domain)
    kwargs = {"mean": torch.from_numpy(coords_stat[0:1,:]),
              "stdev": torch.from_numpy(coords_stat[1:2,:])}
    model = navier_stokes_architecture(layers,**kwargs)
    model.to(device)
    print(model)
    print(model.mean)
    print(model.stdev)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    model.apply(Xavier_initialization)
    return model

def sampling(domain, n_dom, n_bc, device):
    x_dm,y_dm  = sample_collocation_points(domain, n_dom)
    x_dm       = x_dm.to(device)
    y_dm       = y_dm.to(device)
    x_dm       = x_dm.requires_grad_(True)
    y_dm       = y_dm.requires_grad_(True)

    x_bc,y_bc  = sample_boundary_points(domain, n_bc)
    x_bc       = x_bc.to(device)
    y_bc       = y_bc.to(device)
    return x_dm,y_dm, x_bc,y_bc

def para_initialize(num, device):
    Lambda = torch.ones(num, 1, device=device) # pressure, boundary, divergence
    Mu = Lambda * 1.
    Bar_v = Lambda * 0.
    return Lambda, Mu, Bar_v

def printing(mu_evol, lambda_evol, constr_evol, object_evol):
    epoch_evol = np.arange(1, len(object_evol)+1).reshape(-1, 1)
    mu_output = np.concatenate((epoch_evol, np.asarray(mu_evol)), axis=1)
    lambda_output = np.concatenate((epoch_evol, np.asarray(lambda_evol)), axis=1)
    constr_output = np.concatenate((epoch_evol, np.asarray(constr_evol)), axis=1)
    object_output = np.concatenate((epoch_evol, np.asarray(object_evol)), axis=1)
    np.savetxt(f"data/{trial}_mu.dat", mu_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_lambda.dat",lambda_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_constr.dat",constr_output, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_object.dat", object_output, fmt="%.6e", delimiter=" ")

def printing_points(x_dm,y_dm, x_bc,y_bc, trial):
    data_dom = torch.cat((x_dm, y_dm), dim=1).cpu().detach().numpy()
    data_bc  = torch.cat((x_bc, y_bc), dim=1).cpu().detach().numpy()
    np.savetxt(f"data/{trial}_dom.dat", data_dom, fmt="%.6e", delimiter=" ")
    np.savetxt(f"data/{trial}_bc.dat", data_bc, fmt="%.6e", delimiter=" ")

class ParaAdapt:
    def __init__(self, zeta, omega, eta, epsilon):
        self.zeta = zeta
        self.omega = omega
        self.eta = eta
        self.epsilon = epsilon


# In[25]:


def voidlist():
    mu_evol = []
    lambda_evol = []
    constr_evol = []
    object_evol = []
    return mu_evol, lambda_evol, constr_evol, object_evol

def collect_metrics(Mu, Lambda, constr, objective,
                    mu_evol, lambda_evol, constr_evol, object_evol):
    mu_evol.append(Mu.cpu().numpy().flatten())
    lambda_evol.append(Lambda.cpu().numpy().flatten())
    constr_evol.append(constr.detach().cpu().numpy().flatten())
    object_evol.append(objective.detach().cpu().numpy().flatten())


# In[17]:


torch.set_default_dtype(torch.float64)
# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)

Re       = 100 
domain   = np.array([[0.,0.],
                    [1.,1.]])
nu = torch.tensor(1/Re,device=device)

epochs          = 2000
optim_change = True
optim_change_epoch = 0

disp            = 10
print_to_consol = True
disp2           = 500

trials          = 5

n_layers        = 4
neurons         = 30
layers = [2]
for _ in range(n_layers):
    layers.append(neurons)
layers.append(1)

#dom_dis = [100, 100]
#bc_dis  = [128,4]
n_dom = 4096
n_bc  = 128 # per side

num_lambda = 3
para_adapt = ParaAdapt(zeta=0.99, omega=0.999,  
                        eta=torch.tensor([[1.],[1.], [0.01]]).to(device), epsilon=1e-16)
methodname = f'capu_lid_flow_Re{Re}_nn{n_layers}_{neurons}'

# In[25]:


for trial in range(1, trials+1):
    print("*"*20 + f' run({trial}) '+"*"*20)
    model = modelling(domain, layers, device)
    x_dm,y_dm, x_bc,y_bc = sampling(domain, n_dom, n_bc, device)
    printing_points(x_dm,y_dm, x_bc,y_bc, trial)
    # sample point to constrain pressure
    x_p,y_p    = torch.tensor([[0.5]],device=device),torch.tensor([[0.0]],device=device)

    optim_change_loop = optim_change
    optim   = torch.optim.Adam(model.parameters(), lr=1e-3)#,max_iter=7,history_size=10)  

    Lambda, Mu, Bar_v = para_initialize(num_lambda, device)
    #mu_max  = torch.ones((2,1), device=device)
    
    mu_evol, lambda_evol, constr_evol, object_evol = voidlist()
    
    previous_loss = torch.tensor(torch.inf).to(device)
    for epoch in tqdm(range(1,epochs+1)):
        if optim_change_loop and epoch > optim_change_epoch:
            optim        = torch.optim.LBFGS(model.parameters(),line_search_fn="strong_wolfe")
            optim_change_loop =  False

        def _closure():
            model.eval()
            momentum_loss,divergence_loss = PDE_opt(model,x_dm,y_dm,nu)
            boundary_loss                 = boundary_opt(model,x_bc,y_bc)
            pressure_loss                 = pressure_opt(model,x_p,y_p)
    
            objective      = momentum_loss
            constr         = torch.cat((pressure_loss, boundary_loss, divergence_loss), dim=0)
            loss           = objective + (Lambda * constr).sum() + 0.5 * (Mu * constr.pow(2)).sum()
            return objective, constr, loss
    
        def closure():
            if torch.is_grad_enabled():
                model.train()
                optim.zero_grad(set_to_none=True)
            _, _, loss = _closure()
            if loss.requires_grad:
                loss.backward()
            return loss
        
        optim.step(closure)
        
        objective, constr, loss = _closure() 
        
        collect_metrics(Mu, Lambda, constr, objective,
                        mu_evol, lambda_evol, constr_evol, object_evol)    
     
        with torch.no_grad():
            Bar_v        = para_adapt.zeta*Bar_v   + (1-para_adapt.zeta)*constr.pow(2)
            if loss >= para_adapt.omega * previous_loss:
                Lambda   += Mu * constr
                Mu = torch.min( torch.max( para_adapt.eta / (torch.sqrt(Bar_v+para_adapt.epsilon)), Mu ), 2*Mu)
            previous_loss = loss.item()
        
        if epoch % disp == 0 and print_to_consol:
            print('epoch = %d, loss = %.3e, objective = %.3e, constraints = %.3e, %.3e, %.3e'%(epoch, loss, objective, constr[0], constr[1], constr[-1]))
        if epoch % disp2 == 0:
            printing(mu_evol, lambda_evol, constr_evol, object_evol)
            torch.save(model.state_dict(),f"models/{methodname}_{trial}_epoch_{epoch}.pt")
    
    torch.save(model.state_dict(),f"{methodname}_{trial}.pt")
    
    
    # ### plotting configuration
    
    
    contour_vel_mag(model, device, trial, methodname)
    
    plot_horiz_vel(model, device, trial, methodname)
    
    plot_vert_vel(model, device, trial, methodname)
    
    plot_update(trial, methodname)

