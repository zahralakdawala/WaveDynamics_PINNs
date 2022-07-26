from benchmark2 import *
from keras.models import load_model
import matplotlib.pyplot as plt
from plotting import *
from fdm.wave1d_matrixForm import *

network_PINNs    = load_model('trained_networks/wave_trained_network_PINNs.h5')
network_data_100 = load_model('trained_networks/wave_trained_network_data_100.h5')
network_data_400 = load_model('trained_networks/wave_trained_network_data.h5')

plot = 1

if plot == 1:
    
    ModelInfo = benchmark(tend = 1.5)

    T = np.linspace(ModelInfo.tstart, ModelInfo.tend, ModelInfo.num_test_samples)
    X = 0.5
    U = ModelInfo.u0
    
    tx = np.stack([T, np.full(T.shape, X)], axis=-1)
    u_pinns = network_PINNs.predict(tx)
    u_data  = network_data_100.predict(tx)
    u_400   = network_data_400.predict(tx)
    
    plt.plot(T, u_pinns, label = 'PINNs')
    plt.plot(T, u_data, label = 'data_100')
    plt.plot(T, u_400, label = 'data_400')
    plt.plot(T, U(0*T + X, T), label = 'reference')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Evolution of velocity at x = 0.5')
    plt.legend()
    #plt.savefig('plots/evolution_plot.png')
    plt.show()
   
if plot == 2:
    
    ModelInfo = benchmark(tend = 0.2)
    
    T = np.round(np.linspace(0, ModelInfo.num_test_samples - 1, 9)).astype(int)
    tarray = np.linspace(ModelInfo.tstart, ModelInfo.tend, ModelInfo.num_test_samples)
    t_cross_sections = tarray[T]
    
    x_flat = np.linspace(xstart, xend, num_test_samples)
    fig = plt.figure(figsize=(9,8))
    fig.suptitle('Comparision', fontsize = 20, y =1.1)
    
    xarray = np.linspace(xstart, xend, num_test_samples-2)
    u_num, u_exact = FDM_1D(xarray, tarray, c = 1, bc_start = 0, bc_end = 0, plotting = False)
    
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(331 + i)
        tx = np.stack([np.full(tarray.shape, t_cs), x_flat], axis=-1)
        
        u_pinns = network_PINNs.predict(tx)
        u_data100 = network_data_100.predict(tx)
        u_data400 = network_data_400.predict(tx)
        
        plt.ylim(-1, 1)
        plt.plot(x_flat, u_num[:, T[i]], label = 'reference')
        plt.plot(x_flat, u_pinns, label = 'PINNs')
        plt.plot(x_flat, u_data100, label = 'data_100')
        plt.plot(x_flat, u_data400, label = 'data_400')
        plt.title('t={}'.format(round(t_cs, 3)))
        plt.xlabel('x (m)')
        plt.ylabel('u(t,x)')
        plt.legend(loc = 4, prop = {'size' : 7.5})

    plt.tight_layout()
    plt.savefig('plots/wave_numerics.png')
    plt.show()

if plot == 3:
    
    ModelInfo = benchmark(tend = 1)
    # predict u(t,x) distribution
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
    
    t_flat = np.linspace(ModelInfo.tstart, ModelInfo.tend, ModelInfo.num_test_samples)
    x_flat = np.linspace(ModelInfo.xstart, ModelInfo.xend, ModelInfo.num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u_pinns = network_PINNs.predict(tx, batch_size=ModelInfo.num_test_samples)
    u_pinns = u_pinns.reshape(t.shape)
           
    u_data  = network_data_400.predict(tx, batch_size=ModelInfo.num_test_samples)
    u_data  = u_data.reshape(t.shape)
    # plot u(t,x) distribution as a color-map
    heat_map_plot([u_num, u_data, u_pinns], ['Numerical Solution', 'Data_400 Solution', 'PINNs Solution' ], t, x, 'colormap', save = False)
    
    # Error Plot 
    heat_map_plot([abs(u_pinns-u_num), abs(u_data - u_num)], ['Error PINNs', 'Error data_400'], t, x, 'error_colormap', save = False)
