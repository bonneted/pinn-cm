import deepxde as dde
import torch
import numpy as np
# import matplotlib.pyplot as plt
import wandb
import argparse
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(f"Using GPU: {torch.cuda.is_available()}")

lmbd = 1.0
mu = 0.5
Q = 4.0

domain = np.array([[0.0, 1.0], [0.0, 1.0]])
geom = dde.geometry.Rectangle([0, 0], [1, 1])


def U_exact(X):
    x, y = X[:,0], X[:,1]
    #displacement U
    Ux = np.cos(2*np.pi*x) * np.sin(np.pi*y)
    Uy = np.sin(np.pi*x) * Q * y**4/4
    return np.hstack((Ux.reshape(-1,1),Uy.reshape(-1,1)))

def E_exact(X):
    #strain E
    x, y = X[:,0], X[:,1]
    Exx = -2*np.pi*np.sin(2*np.pi*x)*np.sin(np.pi*y)
    Eyy = np.sin(np.pi*x)*Q*y**3
    Exy = 0.5*(np.pi*np.cos(2*np.pi*x)*np.cos(np.pi*y) + np.pi*np.cos(np.pi*x)*Q*y**4/4)
    return (Exx, Eyy, Exy)

def S_exact(X):
    #stress S
    Sxx = (lmbd + 2*mu) * E_exact(X)[0] + lmbd * E_exact(X)[1]
    Syy = (lmbd + 2*mu) * E_exact(X)[1] + lmbd * E_exact(X)[0]
    Sxy = 2*mu*E_exact(X)[2]
    return (Sxx, Syy, Sxy)

# plotting utilities

# def pcolor_plot(AX, X, Y, C, title,colormap="jet",**kwargs):
#     ## plot the pcolor plot of the given data C on the given axis AX with the given title and optional colorbar limits cmin and cmax
#     if len(kwargs) == 0:
#         im = AX.pcolor(X, Y, C, cmap=colormap,shading='auto')
#     else:
#         cmin = kwargs["cmin"]
#         cmax = kwargs["cmax"]
#         im = AX.pcolor(X, Y, C, cmap=colormap, vmin=cmin, vmax=cmax,shading='auto')
#     AX.axis("equal")
#     AX.axis("off")
#     AX.set_title(title, fontsize=14)
#     return im


# def plot_field(domain,model,output_func=None,V_exact=None,plot_diff=False,n_points=10000,fields_name=None):

#     X = np.linspace(domain[0][0], domain[0][1], int(np.sqrt(n_points)))
#     Y = np.linspace(domain[1][0], domain[1][1], int(np.sqrt(n_points)))
#     Xgrid, Ygrid = np.meshgrid(X, Y)
#     Xinput = np.hstack((Xgrid.reshape(-1, 1), Ygrid.reshape(-1, 1)))

#     plotify = lambda x: x.reshape(Xgrid.shape)

#     if output_func is None:
#         V_nn = model.predict(Xinput)
#     else:
#         V_nn = model.predict(Xinput, operator=output_func)

#     V_nn = [plotify(V) for V in V_nn]

#     n_fields = len(V_nn) if type(V_nn) is list else 1
#     n_plot = 1

#     if fields_name is None:
#         fields_name = V_exact.__name__.replace('_exact','') if V_exact is not None else 'V'

#     coord = ["_x","_y","_xy"] if n_fields > 1 else [""]
#     fields_name = [fields_name + coord[i] for i in range(n_fields)]

#     if V_exact is not None:
#         V_exact = V_exact(Xinput)
#         V_exact = [plotify(V) for V in V_exact]
#         n_plot = 3 if plot_diff else 2
        
#     fig, ax = plt.subplots(n_fields, n_plot, figsize=(4*n_plot, 3*n_fields), dpi=200)

#     for i in range(n_fields):

#         if V_exact is not None:

#             cmax = max(V_nn[i].max(), V_exact[i].max())
#             cmin = min(V_nn[i].min(), V_exact[i].min())

#             im1 = pcolor_plot(ax[i][0], Xgrid, Ygrid, V_exact[i], f"{fields_name[i]}*",colormap="jet", cmin=cmin, cmax=cmax)
#             im2 = pcolor_plot(ax[i][1], Xgrid, Ygrid, V_nn[i], f"{fields_name[i]}_nn",colormap="jet", cmin=cmin, cmax=cmax)

#             fig.colorbar(im1, ax=ax[i][1])
#         else:
#             im1 = pcolor_plot(ax[i], Xgrid, Ygrid, V_nn[i], f"{fields_name[i]}_nn")
#             fig.colorbar(im1, ax=ax[i])

#         if plot_diff:
#             diff = V_nn[i] - V_exact[i]
#             abs_diff = np.abs(diff)
#             cmax = abs_diff.max() if diff.max() > 0 else 0
#             cmin = -abs_diff.max() if diff.min() < 0 else 0
#             im3 = pcolor_plot(ax[i][2], Xgrid, Ygrid,diff, f"{fields_name[i]}_nn - {fields_name[i]}*",colormap="jet", cmin=cmin, cmax=cmax)
#             fig.colorbar(im3, ax=ax[i][2])
#             ax[i][2].text(1.075,0.5,f"mean($\mid${fields_name[i]}_nn - {fields_name[i]}*$\mid$): {np.mean(abs_diff):.2e}", fontsize=6,ha = "center",rotation = "vertical",rotation_mode = "anchor")

#     return fig


def bodyf(X, mu, lmbd):
    #body force
    x, y = X[:,0:1], X[:,1:2]
    fx = lmbd*(4*torch.pi**2*torch.cos(2*torch.pi*x)*torch.sin(torch.pi*y)-torch.pi*torch.cos(torch.pi*x)*Q*y**3) \
        + mu*(9*torch.pi**2*torch.cos(2*torch.pi*x)*torch.sin(torch.pi*y) - torch.pi*torch.cos(torch.pi*x)*Q*y**3)
    fy = lmbd*(-3*torch.sin(torch.pi*x)*Q*y**2 + 2*torch.pi**2*torch.sin(2*torch.pi*x)*torch.cos(torch.pi*y)) \
        + mu*(-6*torch.sin(torch.pi*x)*Q*y**2 + 2*torch.pi**2*torch.sin(2*torch.pi*x)*torch.cos(torch.pi*y) + (torch.pi**2*torch.sin(torch.pi*x)*Q*y**4)/4)
    return (fx, fy)


def E_nn(X,U):
    #calculate the strain given the displacement
    # u is the output of the NN, a tensor of shape (N,2)
    Exx = dde.grad.jacobian(U, X, i=0, j=0)
    Eyy = dde.grad.jacobian(U, X, i=1, j=1)
    Exy = 0.5 * (dde.grad.jacobian(U, X, i=1, j=0) +dde.grad.jacobian(U, X, i=0, j=1))
    return Exx, Eyy, Exy

def S_nn(E):
    #calculate the stress given the strain
    Sxx = (2 * mu + lmbd) * E[0] + lmbd * E[1]
    Syy = (2 * mu + lmbd) * E[1] + lmbd * E[0] 
    Sxy = 2 * mu * E[2]
    return Sxx, Syy, Sxy

def PDE(X,S):
    """
    the PDE of the problem (momentum balance)
    X is the spatial coordinate of shape (N,2) (x,y) 
    S is the stress tensor of shape (N,3) (Sxx, Syy, Sxy)
    """

    Sxx, Syy, Sxy = S
    S = torch.stack((Sxx.reshape(-1,1), Syy.reshape(-1,1), Sxy.reshape(-1,1)), axis=1)
        
    dSxx_x = dde.grad.jacobian(S, X, i=0, j=0)
    dSyy_y = dde.grad.jacobian(S, X, i=1, j=1)
    dSxy_x = dde.grad.jacobian(S, X, i=2, j=0)
    dSxy_y = dde.grad.jacobian(S, X, i=2, j=1)
    
    fx, fy = bodyf(X, mu, lmbd)

    momentum_x = dSxx_x + dSxy_y + fx
    momentum_y = dSyy_y + dSxy_x + fy
    
    return [momentum_x, momentum_y]

def E_potential(U,E,S,bodyf):
    """
    calculate the potential energy of the system
    U is the displacement tensor of shape (N,2) U[:,0] = Ux, U[:,1] = Uy
    E is the strain of list of length 3 (Exx, Eyy, Exy)
    S is the stress of list of length 3 (Sxx, Syy, Sxy)
    bodyf is the body force of list of length 2 (fx, fy)
    """
    strain_energy = 0.5 * (S[0]*E[0] + S[1]*E[1] + 2*S[2]*E[2])
    bodyf_work = U[:,0:1] * bodyf[0] + U[:,1:2] * bodyf[1]
    E_pot = strain_energy - bodyf_work
    return E_pot

def Material_error(E,S,lmbd,mu):
    """
    calculate the material behavior error
    E is the strain of list of length 3 (Exx, Eyy, Exy)
    S is the stress of list of length 3 (Sxx, Syy, Sxy)
    lmbd and mu are the material properties for linear elasticity
    """
    Sxx, Syy, Sxy = S
    
    Sxx_pred = (2 * mu + lmbd) * E[0] + lmbd * E[1]
    Syy_pred = (2 * mu + lmbd) * E[1] + lmbd * E[0] 
    Sxy_pred = 2 * mu * E[2]

    Material_error = torch.square(Sxx - Sxx_pred) + torch.square(Syy - Syy_pred) + torch.square(Sxy - Sxy_pred)
    return Material_error


#Unet : displacement u_x and u_y are the output of the network
def PDE_Unet(x,net_output):
    """"
    x: input tensor of shape (N,2), the spatial coordinates x and y
    u: output tensor of shape (N,2), the displacement u_x and u_y
    return: the PDE loss
    """
    E = E_nn(x,net_output)
    S = S_nn(E)
    pde = PDE(x,S)
    return pde

def Epot_Unet(x,net_output):
    """"
    x: input tensor of shape (N,2), the spatial coordinates x and y
    u: output tensor of shape (N,2), the displacement u_x and u_y
    return: the potential energy
    """
    E = E_nn(x,net_output)
    S = S_nn(E)
    bodyf_val = bodyf(x, mu, lmbd)
    E_pot = E_potential(net_output,E,S,bodyf_val)
    return [E_pot]

#USnet : displacement u_x, u_y, and stress S_xx, S_yy, S_xy are the output of the network

def PDE_USnet(x,net_output):
    """"
    x: input tensor of shape (N,2), the spatial coordinates x and y
    net_output: output tensor of shape (N,5), the displacement u_x, u_y, the strain E_xx, E_yy, E_xy
    return: the PDE associated with the network
    """
    S = net_output[:,2], net_output[:,3], net_output[:,4]
    pde = PDE(x,S)
    return pde

def Epot_USnet(x,net_output):
    """
    x: input tensor of shape (N,2), the spatial coordinates x and y
    u: output tensor of shape (N,2), the displacement u_x and u_y
    return: the potential energy associated with the network
    """
    U = torch.hstack((net_output[:,0].reshape(-1,1),net_output[:,1].reshape(-1,1)))
    E = E_nn(x,U)
    S = S_nn(E)
    bodyf_val = bodyf(x, mu, lmbd)
    E_pot = E_potential(U,E,S,bodyf_val)
    return [E_pot]

def MaterialError_USnet(x,net_output):

    U = torch.hstack((net_output[:,0].reshape(-1,1),net_output[:,1].reshape(-1,1)))
    E = E_nn(x,U)
    S = net_output[:,2].reshape(-1,1), net_output[:,3].reshape(-1,1), net_output[:,4].reshape(-1,1)

    return [Material_error(E,S,lmbd,mu)]


#Hard boundary conditions for Unet and USnet (for Unet, hard BCs are applied to displacement only -> soft BCs must be additionnaly applied to stress)
def HardBC_Unet(x,net_output):  
    Ux = net_output[:,0]*x[:,1]*(1-x[:,1])
    Uy = net_output[:,1]*x[:,0]*(1-x[:,0])*x[:,1]
    return torch.hstack((Ux.reshape(-1,1),Uy.reshape(-1,1))) 

def HardBC_USnet(x,net_output):
    Ux = net_output[:,0]*x[:,1]*(1-x[:,1])
    Uy = net_output[:,1]*x[:,0]*(1-x[:,0])*x[:,1]

    Sxx = net_output[:,2]*x[:,0]*(1-x[:,0])
    Syy = net_output[:,3]*(1-x[:,1]) + (lmbd + 2*mu)*Q*torch.sin(torch.pi*x[:,0])
    Sxy = net_output[:,4] 
    return torch.hstack((Ux.reshape(-1,1),Uy.reshape(-1,1),Sxx.reshape(-1,1),Syy.reshape(-1,1),Sxy.reshape(-1,1))) 

#Soft boundary conditions
def boundary_Ux(x,_):
    return np.isclose(x[1], 0) or np.isclose(x[1], 1)

def boundary_Uy(x,_):
    return np.isclose(x[0], 0) or np.isclose(x[0], 1) or np.isclose(x[1], 0)

def boundary_Sxx(x,_):
    return np.isclose(x[0], 0) or np.isclose(x[0], 1)

def boundary_Syy(x,_):
    return np.isclose(x[1], 1)

def BCvalue_Syy(x):
    return (lmbd + 2*mu)*Q*torch.sin(torch.pi*x[:,0])

def Sxx_Unet(inputs, outputs, X):
    E_pred = E_nn(inputs,outputs)
    S_pred = S_nn(E_pred)
    Sxx_pred = S_pred[0]
    return torch.square(Sxx_pred)

def Syy_Unet(inputs, outputs, X):
    E_pred = E_nn(inputs,outputs)
    S_pred = S_nn(E_pred)
    Syy_pred = S_pred[1].squeeze()
    return torch.square(Syy_pred - BCvalue_Syy(inputs))




def loss_and_bc_setup(geom,net,net_type,bc_type,loss_type):
    bc_Ux = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_Ux, component=0)
    bc_Uy = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_Uy, component=1)

    if net_type == 'Unet':
        bc_Sxx = dde.icbc.OperatorBC(geom, Sxx_Unet, boundary_Sxx)
        bc_Syy = dde.icbc.OperatorBC(geom, Syy_Unet, boundary_Syy)
        if bc_type == 'soft':
            bc = [bc_Ux,bc_Uy,bc_Sxx,bc_Syy]
        elif bc_type == 'hard':
            bc = [bc_Sxx,bc_Syy] #Soft BCs for stress
            net.apply_output_transform(HardBC_Unet)
        pde_net = PDE_Unet
        energy_net = Epot_Unet
        mat_net = lambda x,net_output: 0 #No material error for Unet architectur (stress is computed from displacement using the constitutive law)
        if loss_type == 'pde':
            total_loss = pde_net
        elif loss_type == 'energy':
            total_loss = energy_net

    elif net_type == 'USnet':
        #BCs  
        if bc_type == 'soft':
            bc_Sxx = dde.NeumannBC(geom, lambda x: 0, boundary_Sxx, component=2)
            bc_Syy = dde.NeumannBC(geom, BCvalue_Syy, boundary_Syy, component=3)
            bc = [bc_Ux,bc_Uy,bc_Sxx,bc_Syy]
        elif bc_type == 'hard':
            bc = []
            net.apply_output_transform(HardBC_USnet)
        #Loss
        pde_net = PDE_USnet
        energy_net = Epot_USnet
        mat_net = MaterialError_USnet
        if loss_type == 'pde':
            phy_loss = pde_net
        elif loss_type == 'energy':
            phy_loss = energy_net
        # Material error is always added to the loss for USnet architecture
        total_loss = lambda x,net_output: phy_loss(x,net_output) + mat_net(x,net_output)

    return net,total_loss,bc,pde_net,energy_net,mat_net

#Unet exact solution
def Unet_exact(x,lib='np'):
    if lib == 'torch':
        cos,sin,pi,hstack = torch.cos,torch.sin,torch.pi,torch.hstack
    elif lib == 'np':
        cos,sin,pi,hstack = np.cos,np.sin,np.pi,np.hstack
    # ground truth displacement
    Ux = cos(2*pi*x[:,0]) * sin(pi*x[:,1])
    Uy = sin(pi*x[:,0]) * Q * x[:,1]**4/4
    return hstack((Ux.reshape(-1,1),Uy.reshape(-1,1))) 

#USnet exact solution
def USnet_exact(x,lib='np'):
    # ground truth output of the network
    if lib == 'torch':
        cos,sin,pi,hstack = torch.cos,torch.sin,torch.pi,torch.hstack
    elif lib == 'np':
        cos,sin,pi,hstack = np.cos,np.sin,np.pi,np.hstack

    Ux = cos(2*pi*x[:,0]) * sin(pi*x[:,1])
    Uy = sin(pi*x[:,0]) * Q * x[:,1]**4/4
    Exx = -2*pi*sin(2*pi*x[:,0])*sin(pi*x[:,1])
    Eyy = sin(pi*x[:,0])*Q*x[:,1]**3
    Exy = 0.5*(pi*cos(2*pi*x[:,0])*cos(pi*x[:,1]) + pi*cos(pi*x[:,0])*Q*x[:,1]**4/4)
    S = S_nn((Exx,Eyy,Exy))
    Sxx, Syy, Sxy = S[0], S[1], S[2]
    return hstack((Ux.reshape(-1,1),Uy.reshape(-1,1),Sxx.reshape(-1,1),Syy.reshape(-1,1),Sxy.reshape(-1,1))) 

def train_PINN(geom,net_paramters,loss_parameters,optimizer_parameters,callbacks = []):
    
    [net_type,n_layers,size_layers,activation] = net_paramters
    [loss_type,num_sample,train_distribution,bc_type] = loss_parameters
    [optimizers,learning_rates,iterations] = optimizer_parameters
    
    net_exact = Unet_exact if net_type == 'Unet' else USnet_exact
    size_output = 2 if net_type == 'Unet' else 5
    num_domain = num_sample**2
    num_boundary = num_sample

    net = dde.nn.FNN([2] + [size_layers]*n_layers + [size_output], activation, 'Glorot uniform')
    net, total_loss, bc, pde_net, energy_net, mat_net = loss_and_bc_setup(geom,net, net_type, bc_type, loss_type)

    data = dde.data.PDE(
        geom,
        total_loss,
        bc,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_test= 100**2,
        train_distribution = train_distribution,
        solution=net_exact,
    )

    model = dde.Model(data, net)
    
    for i,optimizer in enumerate(optimizers) :    
        if optimizer == "adam":
            model.compile("adam", lr=learning_rates[i], metrics=["l2 relative error"])
            losshistory, train_state  = model.train(iterations=iterations[i], display_every=200,callbacks = callbacks,model_save_path=os.path.join(wandb.run.dir, "model"))
        else:
            model.compile("L-BFGS", metrics=["l2 relative error"])
            losshistory, train_state  = model.train(callbacks = callbacks,model_save_path=os.path.join(wandb.run.dir, "model"))
    return model, losshistory, train_state

def log_run(model,config,losshistory,log_plot = False):

    steps = np.array(losshistory.steps).squeeze() 
    losses_train = np.array(losshistory.loss_train).squeeze().sum(axis=1)
    losses_test = np.array(losshistory.loss_test).squeeze().sum(axis=1)
    metrics = np.array(losshistory.metrics_test).squeeze()

    for i in range(len(steps)):
        wandb.log({"steps": steps[i], "loss_train": losses_train[i], "loss_test": losses_test[i], "accuracy_metric": metrics[i]})

#     if log_plot:
#         U_output = lambda x, ouput: (ouput[:,0], ouput[:,1]) 
#         S_output = lambda x, ouput: S_nn(E_nn(x,ouput)) if config.net_type == 'Unet' else (ouput[:,2], ouput[:,3], ouput[:,4]) 

#         U_field = plot_field(domain,model,output_func=U_output,V_exact=lambda x: (U_exact(x)[:,0], U_exact(x)[:,1]),plot_diff=True,fields_name="U")
#         Eps_field = plot_field(domain,model,output_func=E_nn,V_exact=E_exact,plot_diff=True)
#         Sig_field =plot_field(domain,model,output_func=S_output,V_exact=S_exact,plot_diff=True)

#         wandb.log({"U_field": wandb.Image(U_field)})
#         wandb.log({"Eps_field": wandb.Image(Eps_field)})
#         wandb.log({"Sig_field": wandb.Image(Sig_field)})

 
def wandb_train(config=None):
    
    dde.optimizers.config.set_LBFGS_options(maxiter=10000)

    with wandb.init(project="HPO-PINN-CM", config=config):
        config = wandb.config
        net_paramters =    [config.net_type,config.n_layers,config.size_layers,config.activation]
        loss_parameters = [config.loss_type,config.num_samples,config.train_distribution,config.bc_type]
        optimizer_parameters = [config.optimizer,config.learning_rate,config.iterations]

        model, losshistory, train_state = train_PINN(geom,net_paramters,loss_parameters,optimizer_parameters)
        log_run(model,config,losshistory,log_plot = False)


if __name__ == "__main__":

    domain = np.array([[0.0, 1.0], [0.0, 1.0]])
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    parser = argparse.ArgumentParser()
    parser.add_argument("--net_type", type=str, default="Unet")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--size_layers", type=int, default=20)
    parser.add_argument("--activation", type=str, default="tanh")

    parser.add_argument("--loss_type", type=str, default="pde")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--train_distribution", type=str, default="uniform")
    parser.add_argument("--bc_type", type=str, default="hard")

    #     parser.add_argument("--optimizer", nargs="+", type=str, default="adam")
    #     parser.add_argument("--learning_rate",nargs="+", type=list)
    #     parser.add_argument("--iterations", nargs="+", type=list)

    config = vars(parser.parse_args())
                                                    
#     config["optimizer"] = ["adam"]
#     config['learning_rate'] = [1e-3]
#     config["iterations"] = [300]

    config["optimizer"] = ["adam","L-BFGS"]
    config['learning_rate'] = [1e-3,None]
    config["iterations"] = [3000,None]

    wandb_train(config)


