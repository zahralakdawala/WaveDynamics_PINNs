import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from keras.models import load_model
from benchmark import *
from lib.network import Network
from lib.optimizer import *
from lib.pinn import PINN

 

if __name__ == '__main__':

    ModelInfo = SWE()
    
    if ModelInfo.train:
        
        # build a core network model
        network = Network().build(layers = ModelInfo.layers)
        network.summary()
        
        # build a PINN model
        pinn = PINN(network).build()
    
        # training input and output
        x_train = ModelInfo.xtrain
        y_train = ModelInfo.ytrain
    
        # train the model using L-BFGS-B algorithm
        lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
        lbfgs.fit()
        
        network.save('trained_networks/SWE_maxiter10000_b'+ str(ModelInfo.benchmark) +'_'+ ModelInfo.mode +'.h5')
        
    else:
        network = load_model('trained_networks/SWE_maxiter10000_b'+ str(ModelInfo.benchmark) +'_' + ModelInfo.mode +'.h5')


    
    # Predictions and plotting
    ModelInfo.heat_map_plot(network)
    ModelInfo.plots(network)
    
