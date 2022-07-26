from benchmark import *
from keras.models import load_model
import matplotlib.pyplot as plt

Model = SWE()
benchmark = Model.benchmark

network_PINNs    = load_model('trained_networks/SWE_maxiter10000_b'+str(benchmark)+'_PINNs.h5')
network_data     = load_model('trained_networks/SWE_maxiter10000_b'+str(benchmark)+'_data.h5')
if benchmark == 2:
    network_dataPINN = load_model('trained_networks/SWE_maxiter10000_b2_dataAndPhysics.h5')

plot = 3
save = True

if plot == 1:
    
    ModelInfo = InputVariables(tend = 0.25)

    T = np.linspace(ModelInfo.tstart, ModelInfo.tend, ModelInfo.num_test_samples)
    X = 0.5
    
    qua = len(ModelInfo.x_num) // 2
    
    tx = np.stack([T, np.full(T.shape, X)], axis=-1)
    h_pinns = network_PINNs.predict(tx)[..., 0]
    h_data  = network_data.predict(tx)[..., 0]
    
    plt.plot(T, h_pinns, label = 'h PINNs')
    plt.plot(T, h_data, label = 'h data')
    if ModelInfo.benchmark == 2:
        h_dPhys = network_dataPINN.predict(tx)[..., 0]
        plt.plot(T, h_dPhys, label = 'h dataAndPhysics')
    plt.plot(ModelInfo.t_num, ModelInfo.h_num[qua, :], label = 'h reference')
    plt.ylim(1.5, 3)
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('Evolution of height at x = 0')
    plt.legend()
    if save:
        plt.savefig('plots/h_evolution_plot_b'+str(ModelInfo.benchmark)+'.png')
    plt.show()
    
    uh_pinns = network_PINNs.predict(tx)[..., 1] * network_PINNs.predict(tx)[..., 0]
    uh_data  = network_data.predict(tx)[..., 1] 
        
    plt.plot(T, uh_pinns, label = 'uh PINNs')
    plt.plot(T, uh_data, label = 'uh data')
    if ModelInfo.benchmark == 2:
        uh_dPhys = network_dataPINN.predict(tx)[..., 1] 
        plt.plot(T, uh_dPhys, label = 'uh dataAndPhysics')
    plt.plot(ModelInfo.t_num, ModelInfo.uh_num[qua, :], label = 'uh reference')
    plt.ylim(-2, 2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Density (m^2/s)')
    plt.title('Evolution of velocity density at x = 0')
    plt.legend()
    if save:
        plt.savefig('plots/uh_evolution_plot_b'+str(ModelInfo.benchmark)+'.png')
    plt.show()


if plot == 2:
    ModelInfo = InputVariables(tend = 0.2)
    
    h_num  = ModelInfo.h_num
    uh_num = ModelInfo.uh_num
    x_num  = ModelInfo.x_num
    t_num  = ModelInfo.t_num
    
    T = np.round(np.linspace(0, len(t_num) - 1, 9)).astype(int)
    t_cross_sections = t_num[T]
    fig = plt.figure(figsize=(9,8))
    fig.suptitle('Comparision of height', fontsize = 20, y =1.1)
    
    minH = np.min(h_num) - 0.5
    maxH = np.max(h_num) + 0.5
    
    uh_array = []

    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(331 + i)
        tx = np.stack([np.full(x_num.shape, t_cs), x_num], axis=-1)

        sol_data = network_data.predict(tx, batch_size=len(tx))
        sol_pinn = network_PINNs.predict(tx, batch_size=len(tx))
                
        h_data   = sol_data[..., 0]
        h_pinn   = sol_pinn[..., 0]
         
        uh_data  = sol_data[..., 1]
        uh_pinn  = sol_pinn[..., 1] * sol_pinn[..., 0]
        
        plt.ylim(minH, maxH)
        plt.plot(x_num, h_num[:, T[i]], label = 'h reference')
        plt.plot(x_num, h_data, label = 'h data')
        plt.plot(x_num, h_pinn, label = 'h PINNs')
        if ModelInfo.benchmark == 2:
            sol_dPhy = network_dataPINN.predict(tx, batch_size=len(tx))
            h_dPhy   = sol_dPhy[..., 0]
            uh_dPhy  = sol_dPhy[..., 1] 
            uh_array.append([uh_data, uh_pinn, uh_dPhy])
            plt.plot(x_num, h_dPhy, label = 'h dataAndPhysics')
        else: 
            uh_array.append([uh_data, uh_pinn])
        plt.title('t={}'.format(round(t_cs, 3)))
        plt.ylabel('h(t,x)')
        plt.xlabel('x (m)')
        plt.legend(loc = 4, prop = {'size' : 6})

    plt.tight_layout()
    
    if save:
        plt.savefig('plots/h_b'+ str(ModelInfo.benchmark)+'.png')
    plt.show()
    
    # FOR UH
    
    fig = plt.figure(figsize=(9,8))
    fig.suptitle('Comparision of velocity density', fontsize = 20, y =1.1)
    
    minUH = np.min(uh_num) - 0.8
    maxUH = np.max(uh_num) + 0.5
    
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(331 + i)
        if ModelInfo.benchmark == 1:
            uh_data, uh_pinn = uh_array[i]
        else:
            uh_data, uh_pinn, uh_dPhy = uh_array[i]
        
        plt.ylim(minUH, maxUH)
        plt.plot(x_num, uh_num[:, T[i]], label = 'uh reference')
        plt.plot(x_num, uh_data, label = 'uh data')
        plt.plot(x_num, uh_pinn, label = 'uh PINNs')
        if ModelInfo.benchmark == 2:
            plt.plot(x_num, uh_dPhy, label = 'uh dataAndPhysics')
        
        plt.title('t={}'.format(round(t_cs, 3)))
        plt.xlabel('x (m)')
        plt.ylabel('uh(t,x)')
        plt.legend(loc = 4, prop = {'size' : 6})

    plt.tight_layout()
    
    if save:
        plt.savefig('plots/uh_b'+ str(ModelInfo.benchmark)+'.png')
    plt.show()


        
if plot == 3:
    
    def heat_map_plot(sol_array, title_array, t, x, titlefile, label_plot, ymin = -1, ymax = 1, save = True):
        fig = plt.figure(figsize=(9,12))
        gs = GridSpec(len(sol_array), 3)

        ymin = min(np.array(sol_array).reshape(-1, ))
        ymax = max(np.array(sol_array).reshape(-1, ))
        
        for i in range(len(sol_array)):
            plt.subplot(gs[i, :])
            plt.title(title_array[i])
            plt.xlabel('t')
            plt.ylabel('x')
            vmin, vmax = ymin, ymax
            plt.pcolormesh(t, x, sol_array[i], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))

            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.set_label('u(t,x)')
            cbar.mappable.set_clim(vmin, vmax)
        plt.tight_layout()
        if save:
            plt.savefig('plots/'+ titlefile + '.png')
        plt.show()
    
    # Starts here
    
    ModelInfo = InputVariables(tend = 0.2)
    
    # create meshgrid coordinates (x, y) for test plots
    h_num  = ModelInfo.h_num
    uh_num = ModelInfo.uh_num
    x_num  = ModelInfo.x_num
    t_num  = ModelInfo.t_num
    
    t, x = np.meshgrid(t_num, x_num)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)

    # predict (psi, p)
    sol_pinn = network_PINNs.predict(tx) 
    sol_data = network_data.predict(tx)
    
    h_pinn, u_pinn = [ sol_pinn[..., i].reshape(t.shape) for i in range(sol_pinn.shape[-1]) ]
    uh_pinn = h_pinn * u_pinn
    
    h_data, uh_data = [ sol_data[..., i].reshape(t.shape) for i in range(sol_data.shape[-1]) ]
    
    if ModelInfo.benchmark == 2:
        sol_dPhy = network_dataPINN.predict(tx)
        h_dPhy, uh_dPhy = [ sol_dPhy[..., i].reshape(t.shape) for i in range(sol_dPhy.shape[-1]) ]
        h_array = [h_num, h_data, h_pinn, h_dPhy]   
        title_array = [' h numerical', 'h data', 'h PINNs', 'h dataAndPhysics']

        uh_array = [uh_num, uh_data, uh_pinn, uh_dPhy]   
        title_array_uh = [' uh numerical', 'uh data', 'uh PINNs', 'uh dataAndPhysics']
        
    else:
        h_array = [h_num, h_data, h_pinn]
        title_array = [' h numerical', 'h data', 'h PINNs']
        uh_array = [uh_num, uh_data, uh_pinn]  
        title_array_uh = [' uh numerical', 'uh data', 'uh PINNs']
        
    heat_map_plot(h_array, title_array, t, x, 'h_colormap_b'+str(ModelInfo.benchmark), 'h(t, x)', save = save)
    heat_map_plot(uh_array, title_array_uh, t, x, 'uh_colormap_b'+str(ModelInfo.benchmark), 'uh(t, x)', save = save)
    heat_map_plot([abs(h_array[i] - h_num) for i in range(1, len(h_array))], title_array[1:], \
                  t, x, 'h_errorplot_b'+str(ModelInfo.benchmark), 'h(t, x)', save = save)
        
    heat_map_plot([abs(uh_array[i] - uh_num) for i in range(1, len(uh_array))], title_array_uh[1:], \
                  t, x, 'uh_errorplot_b'+str(ModelInfo.benchmark), 'uh(t, x)', save = save)
