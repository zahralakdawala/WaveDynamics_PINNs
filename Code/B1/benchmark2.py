import numpy as np
import tensorflow as tf
from numpy import pi
from fdm.wave1d_matrixForm import *



# Exact Solution 
class benchmark:
    def __init__(self, xstart = 0, xend = 1, tstart = 0, tend = 1):      
        # Variables
        self.xstart = xstart
        self.xend   = xend
        self.Nx     = 100
        self.mode   = 'dataAndPhysics'  # 'PINNs' or 'data'
        
        
        self.tstart = tstart
        # Make the value of tstart = 0.8 when training the network and 1 when testing
        self.tend   = tend
        self.Nt     = 100

        self.num_train_samples = 100
        self.num_test_samples  = 100
        
        self.train = True
        #False if you have a saved network, True to activate training step

        self.layers = [32, 32, 32, 32]

        # Spatial steps
        self.x = np.linspace(self.xstart, self.xend, self.Nx)
        # Time steps
        self.t = np.linspace(self.tstart, self.tend, self.Nt)

        # Training Input 
        self.tx_eqn = np.random.rand(self.num_train_samples, 2)
        self.tx_eqn[..., 0] = self.tx_eqn[..., 0]            # t = 0 ~ 0.8
        self.tx_eqn[..., 1] = self.tx_eqn[..., 1]                # x = 0 ~ +1
        
        self.tx_ini = np.random.rand(self.num_train_samples, 2)
        self.tx_ini[..., 0] = 0                                  # t = 0
        self.tx_ini[..., 1] = self.tx_ini[..., 1]                # x = 0 ~ +1
        
        self.tx_bnd = np.random.rand(self.num_train_samples, 2)
        self.tx_bnd[..., 0] = self.tx_bnd[..., 0]                # t =  0 ~ 0.8
        self.tx_bnd[..., 1] = np.round(self.tx_bnd[..., 1])      # x = 0 or +1
        
        if self.mode == 'PINNs':
            self.xtrain = [self.tx_eqn, self.tx_ini, self.tx_bnd]
        elif self.mode == 'data':
            self.xtrain = [self.tx_eqn, self.tx_bnd]
        elif self.mode == 'dataAndPhysics':
            self.xtrain = [self.tx_eqn, self.tx_ini, self.tx_bnd]
        
        
        
        #Training Output
        self.u_zero = np.zeros((self.num_train_samples, 1))
        self.u_ini = self.u0(tf.constant(self.tx_ini), noise = True).numpy()
        self.du_dt_ini = self.du0_dt(tf.constant(self.tx_ini)).numpy()
        
        if self.mode == 'PINNs':
            self.ytrain = [self.u_zero, self.u_ini[:, 1].reshape(-1, 1), self.du_dt_ini, self.u_zero]
        elif self.mode == 'data':
            self.u_sol = self.u0(self.tx_eqn[...,1], self.tx_eqn[...,0]).numpy()
            self.ytrain = [self.u_sol.reshape(-1, 1), self.u_zero]
        elif self.mode == 'dataAndPhysics':
            self.u_sol = self.u0(self.tx_eqn[...,1], self.tx_eqn[...,0]).numpy()
            self.ytrain = [self.u_zero, self.u_sol.reshape(-1, 1), self.u_ini[:, 1].reshape(-1, 1), self.du_dt_ini, self.u_zero]
        
  
                
    def u0(self, x, t = 0, c=1, noise = False):
        """
        Exact solution of wave equation
        
        Args:
            tx: variables (t, x) as tf.Tensor.
            c: wave velocity.
            k: wave number.
            sd: standard deviation.
    
        Returns:
            u(t, x) as tf.Tensor.
        """
        pi = tf.constant(np.pi, dtype = 'float64')
        
        u =  0.5*tf.sin(pi*x)*tf.cos(pi*t) + (1/3)*tf.sin(3*pi*x)*tf.sin(3*pi*t) 
        
        if noise :    
            u += np.random.randn(u.shape[0], u.shape[1]) * 1e-2
        
        return u


    # Derivative of Exact Solution
    
    # Returns tensor
    def du0_dt(self, tx):
        """
        First derivative of wave equation
        
        Args:
            tx: variables (t, x) as tf.Tensor.
    
        Returns:
            du(t, x)/dt as tf.Tensor.
        """
        t = tx[..., 0, None]
        x = tx[..., 1, None]
       
        return (-1/2)*pi*tf.sin(pi*t)*tf.sin(pi*x)+pi*tf.cos(3*pi*t)*tf.sin(3*pi*x)
    
    # Return numpy array
    def dt_I(self, x, t=0):
        return (-1/2)*pi*np.sin(pi*t)*np.sin(pi*x)+pi*np.cos(3*pi*t)*np.sin(3*pi*x)
    

    







