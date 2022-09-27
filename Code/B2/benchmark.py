import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.SWE1D import *


            

class InputVariables:
    def __init__(self, num_train = 200, num_test = 200, xstart = 0, \
                 xend = 1, tstart = 0, tend = 0.15, layers = 10, \
                 nodes = 30, train = True, mode = 'dataAndPhysics',\
                 benchmark = 1):
        
        # number of training samples
        self.num_train_samples = num_train
        # number of test samples
        self.num_test_samples = num_test
        
        self.xstart = xstart
        self.xend   = xend
        self.tstart = tstart
        self.tend   = tend
    
        self.layers = [nodes] * layers

        self.g = 9.8
        
        self.train = train
        self.mode = mode
        
        self.benchmark = benchmark
        
        h_num, uh_num, x_num, t_num = shallow_water_1d_test(self.benchmark, self.xend, self.tend)
        
        self.h_num = h_num
        self.uh_num = uh_num
        self.x_num = x_num
        self.t_num = t_num
            
        if self.mode in ['dataAndPhysics', 'data']:
            sign = -1
            if self.mode == 'data':
                sign = 1
            idx = (np.arange(len(t_num))[::20])[1:]

            x, t, hsol, uhsol = [], [], [], []
            
            for i in idx:
                hsol.append(h_num[:, i])
                uhsol.append(uh_num[:, i])
                x.append(x_num)
                t.append([t_num[i]] * len(x_num))

            self.tsol  = np.array(t).reshape(-1, 1)   
            self.xsol  = np.array(x).reshape(-1, 1)
            self.hsol  = np.array(hsol).reshape(-1, 1)
            self.uhsol = np.array(uhsol).reshape(-1, 1)
            
            self.num_train_samples = len(self.tsol)
            print(self.num_train_samples)
        
#InputVariables()

class TrainingInput(InputVariables):
    def __init__(self):
        InputVariables.__init__(self)
        
        self.tx_eqn = np.random.rand(self.num_train_samples, 2)
        self.tx_eqn[..., 0] = self.tend*self.tx_eqn[..., 0]
        self.tx_eqn[..., 1] = self.xend*self.tx_eqn[..., 1]
        
        self.tx_ini = np.random.rand(self.num_train_samples, 2)
        self.tx_ini[..., 0] = 0
        self.tx_ini[..., 1] = self.xend*self.tx_ini[..., 1]

        self.tx_bnd = np.random.rand(self.num_train_samples, 2)
        self.tx_bnd[..., 0] = self.tend*self.tx_bnd[..., 0]
        self.tx_bnd[..., 1] = self.xend*np.round(self.tx_bnd[..., 1])

        if self.mode in ['dataAndPhysics', 'data']:
            self.tx_sol = np.concatenate([self.tsol, self.xsol], axis = 1)
        
    def get_xtrain(self):
        if self.mode == 'dataAndPhysics':
            return [self.tx_eqn, self.tx_ini, self.tx_sol]
        elif self.mode == 'PINNs':
            return [self.tx_eqn, self.tx_ini]
        elif self.mode == 'data':
            return [self.tx_sol]

    

class TrainingOutput(TrainingInput):
    def __init__(self):
        TrainingInput.__init__(self)
        # create training output
        x_flat  = np.linspace(self.xstart, self.xend, self.num_train_samples)
        t_flat  = np.linspace(self.tstart, self.tend, self.num_train_samples)
        self.zeros   = np.zeros((self.num_train_samples, 2))
    
        self.hu_ini  = self.initial_conditions(self.tx_ini[..., 1], noise = True)

        if self.mode in ['dataAndPhysics', 'data']:
            self.hu_sol  = np.concatenate([self.hsol, self.uhsol], axis = 1)

    def get_ytrain(self):
        if self.mode == 'dataAndPhysics':

            return [self.zeros, self.hu_ini, self.hu_sol]
        elif self.mode == 'PINNs':
            return [self.zeros, self.hu_ini]
        elif self.mode == 'data':
            return [self.hu_sol]
    
    def damBreak(self, x, h1 = 1, h2 = 0.5):
        val = []
        step1 = 0.45*(self.xend - self.xstart)
        step2 = 0.55*(self.xend - self.xstart)
        
            
        for i in x:
            if i < step1:
                val.append(h1 + 10e-3*np.random.randn())
            elif i > step2:
                val.append(h2 + 10e-3*np.random.randn())
            else:
                m = (h2-h1)/(step2-step1)
                c = h2 - m*step2
                val.append( m*i + c + 10e-3*np.random.randn())
                
        return val
        
    def initial_conditions (self, x, t=0, noise = False):
        nx = self.num_train_samples
        h_uh = np.zeros((nx, 2))
        
        if self.benchmark == 1:
            if noise:
                random = np.random.rand(len(x))*10e-3
            else:
                random = 0
                
            h_uh[:, 0] +=  0.5*np.sin(np.pi*x) + random
        elif self.benchmark == 2:
            h_uh[:, 0] += 2.0 + 0.1*np.sin ( 2.0 * np.pi * x )
        elif self.benchmark == 3: #Weak Shock
            h_uh[:, 0] += self.damBreak(x)
        elif self.benchmark == 4: #Mild Shock
            h_uh[:, 0] += self.damBreak(x, h1 = 1, h2 = 0.1)
        elif self.benchmark == 5: #Strong shock
            h_uh[:, 0] += self.damBreak(x, h1 = 1, h2 = 0.02)


        return h_uh


class SWE(TrainingOutput):
    
    def __init__(self):  
        TrainingOutput.__init__(self)
        self.xtrain = self.get_xtrain()
        self.ytrain = self.get_ytrain()
       
    
    def heat_map_plot(self, network, ymin = -1, ymax = 4, save = True):
        
        # create meshgrid coordinates (x, y) for test plots
        h_num  = self.h_num
        uh_num = self.uh_num
        x_num  = self.x_num
        t_num  = self.t_num
        
        t, x = np.meshgrid(t_num, x_num)
        tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    
        # predict (psi, p)
        h_uh = network.predict(tx) #, batch_size=len(tx))
        h, u = [ h_uh[..., i].reshape(t.shape) for i in range(h_uh.shape[-1]) ]
        
        sol_array = [h, h_num]                         
        title_array = ['h '+self.mode, ' h numerical']
        
        fig = plt.figure(figsize=(7, 6))
        gs = GridSpec(len(sol_array), 3)
        
        ymin = min(np.array(sol_array).reshape(-1,))
        ymax = max(np.array(sol_array).reshape(-1,))
        
        for i in range(len(sol_array)):
            plt.subplot(gs[i, :])
            plt.title(title_array[i])
            plt.xlabel('t')
            plt.ylabel('x')

            
            vmin, vmax = ymin, ymax
            plt.pcolormesh(t, x, sol_array[i], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))

            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.set_label('h(t,x)')
            cbar.mappable.set_clim(vmin, vmax)
        plt.tight_layout()
        if save:
            plt.savefig('plots/b'+ str(self.benchmark)+'_h_'+ (self.mode).lower()+'_numerical.png')
        plt.show()
        
        if self.mode == 'PINNs':
            sol_array = [u*h, uh_num]   
        else:
            sol_array = [u, uh_num]   
        title_array = ['uh '+self.mode, 'uh numerical']
        
        fig = plt.figure(figsize=(7, 6))
        gs = GridSpec(len(sol_array), 3)
        
        ymin = min(np.array(sol_array).reshape(-1,))
        ymax = max(np.array(sol_array).reshape(-1,))
            
        for i in range(len(sol_array)):
            plt.subplot(gs[i, :])
            plt.title(title_array[i])
            plt.xlabel('t')
            plt.ylabel('x')

            vmin, vmax = ymin, ymax
            plt.pcolormesh(t, x, sol_array[i], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))

            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.set_label('uh(t,x)')
            cbar.mappable.set_clim(vmin, vmax)
        plt.tight_layout()
        plt.legend()
        if save:
            plt.savefig('plots/b'+ str(self.benchmark)+'_uh_'+ (self.mode).lower()+'_numerical.png')
        plt.show()
        
        if self.mode == 'PINNs':
            sol_array = [abs(h - h_num), abs(uh_num - u*h)]   
        else:
            sol_array = [abs(h - h_num), abs(uh_num - u)]           
        title_array = ['h error', 'uh error']
        
        fig = plt.figure(figsize=(7, 6))
        gs = GridSpec(len(sol_array), 3)
        
        ymin = min(np.array(sol_array).reshape(-1,))
        ymax = max(np.array(sol_array).reshape(-1,))
        
        for i in range(len(sol_array)):
            plt.subplot(gs[i, :])
            plt.title(title_array[i])
            plt.xlabel('t')
            plt.ylabel('x')

            vmin, vmax = ymin, ymax
            plt.pcolormesh(t, x, sol_array[i], cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))

            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.set_label('error')
            cbar.mappable.set_clim(vmin, vmax)
        plt.tight_layout()
        if save:
            plt.savefig('plots/b'+ str(self.benchmark)+ '_' + self.mode +'_h_uh_errors.png')
        plt.show()
        
    def plots(self, network, save = True):
        
        #h_num, uh_num, x_num, t_num = shallow_water_1d_test ( )
        h_num  = self.h_num
        uh_num = self.uh_num
        x_num  = self.x_num
        t_num  = self.t_num
        
        T = np.round(np.linspace(0, len(t_num) - 1, 9)).astype(int)
        t_cross_sections = t_num[T]
        fig = plt.figure(figsize=(7,6))
        fig.suptitle('Comparision', fontsize = 20, y =1.1)
        
        minH = np.min(h_num)
        maxH = np.max(h_num)
        print("minH, maxH", minH, maxH)
        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(331 + i)
            tx = np.stack([np.full(x_num.shape, t_cs), x_num], axis=-1)
    
            sol = network.predict(tx, batch_size=len(tx))
            h  = sol[..., 0]
            uh = sol[..., 1]
            plt.ylim(minH, maxH)
            plt.plot(x_num, h_num[:, T[i]], label = 'numerical')
            plt.plot(x_num, h, label = self.mode)
            plt.title('t={}'.format(round(t_cs, 3)))
    
            plt.ylabel('h(t,x)')
            plt.legend()
    
        plt.tight_layout()
        if save:
            plt.savefig('plots/h_bc'+ str(self.benchmark)+'_'+self.mode+'.png')
        plt.show()
        
        fig = plt.figure(figsize=(7,6))
        fig.suptitle('Comparision', fontsize = 20, y =1.1)
    
        uh_min = np.min(uh_num)
        uh_max=np.max(uh_num)
        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(331 + i)
            tx = np.stack([np.full(x_num.shape, t_cs), x_num], axis=-1)
            sol = network.predict(tx, batch_size=len(tx))
            h  = sol[..., 0]
            u  = sol[..., 1]
            plt.ylim(uh_min, uh_max)
     
            plt.plot(x_num, uh_num[:, T[i]], label = 'numerical')
            if self.mode == 'PINNs':
                plt.plot(x_num, u*h, label = 'pinns')
            else:
                plt.plot(x_num, u, label = self.mode)
            plt.title('t={}'.format(round(t_cs, 3)))
    
            plt.ylabel('uh(t,x)')
            plt.legend(loc = 1, prop = {'size' : 6})
    
        plt.tight_layout()
        if save:
            plt.savefig('plots/uh_bc'+ str(self.benchmark)+'_'+self.mode+'.png')
        plt.show()