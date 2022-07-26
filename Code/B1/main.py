import lib.tf_silent
from lib.network import Network
from lib.optimizer import L_BFGS_B
from pinn import PINN
import numpy as np
from numpy import pi
from numpy import linalg as la
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from benchmark2 import *
from plotting import *
from keras.models import load_model
from fdm.wave1d_matrixForm import *



ModelInfo = benchmark()

u0     = ModelInfo.u0
du0_dt = ModelInfo.du0_dt

xstart = ModelInfo.xstart
xend   = ModelInfo.xend
tstart = ModelInfo.tstart
tend   = ModelInfo.tend

# number of training samples
num_train_samples = ModelInfo.num_train_samples

# number of test samples
num_test_samples  = ModelInfo.num_test_samples


# Numerical Solution
xarray = np.linspace(xstart, xend, num_test_samples-2)
tarray = np.linspace(tstart, tend, num_test_samples)


u_num, u_exact = FDM_1D(xarray, tarray, c = 1, bc_start = 0, bc_end = 0, plotting = False)


if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """
    
    if ModelInfo.train:
        # build a core network model
        network = Network.build(layers = ModelInfo.layers)
        network.summary()
        
        # build a PINN model
        pinn = PINN(network).build()
        pinn.summary()
        
        # create training input
        tx_eqn = ModelInfo.tx_eqn
        tx_ini = ModelInfo.tx_ini
        tx_bnd = ModelInfo.tx_bnd
        
        # train the model using L-BFGS-B algorithm
        x_train = ModelInfo.xtrain
        y_train = ModelInfo.ytrain
        lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
        lbfgs.fit()
        
        network.save('trained_networks/wave_trained_network_' +ModelInfo.mode +'_100.h5')  # creates a HDF5 file 'my_model.h5'
        
    else:
        # returns a compiled model
        network = load_model('trained_networks/wave_trained_network_'+ ModelInfo.mode +'_100.h5')
        network.summary()
        
    # predict u(t,x) distribution
    t_flat = np.linspace(tstart, tend, num_test_samples)
    x_flat = np.linspace(xstart, xend, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)
           

    # plot u(t,x) distribution as a color-map
    heat_map_plot([u0(x, t), u, u_num], ['Exact Solution', ModelInfo.mode +' Solution', 'Numerical Solution'], t, x, 'pinns_solution_heatmap_' + ModelInfo.mode, save = False)
    
    # Error Plot 
    heat_map_plot([abs(u-u_num)], ['Error Plot'], t, x, 'error_colormap_'+ ModelInfo.mode, save = False)
     
    # plot u(t=const, x) cross-sections
    
    '''
        This plots predicted solution of PINNs along with Numerical solution
        and its corresponding error at each time step
    '''
    T = np.round(np.linspace(0, num_test_samples - 1, 9)).astype(int)
    t_cross_sections = tarray[T]
    fig = plt.figure(figsize=(7,6))
    fig.suptitle('Comparision', fontsize = 20, y =1.1)
    
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(331 + i)
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = network.predict(tx, batch_size=num_test_samples)
        plt.ylim(-1, 1)
        plt.plot(x_flat, u_num[:, T[i]], label = 'Numerical')
        plt.plot(x_flat, u_exact[:, T[i]], label = 'Exact')
        plt.plot(x_flat, u, label = ModelInfo.mode)
        plt.title('t={}'.format(round(t_cs, 3)))
        plt.xlabel('relative error : %s' %(round(la.norm(u - u_exact[:, T[i]], ord=2)/(la.norm(u_exact, ord =2)),3)))
        plt.ylabel('u(t,x)')
        plt.legend(loc = 1, prop = {'size' : 6})

    plt.tight_layout()
    plt.show()

    







