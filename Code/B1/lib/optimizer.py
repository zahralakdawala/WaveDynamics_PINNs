import scipy.optimize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from benchmark2 import *

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.

    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10 for extremely high accuracy.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: logging metrics.
        progbar: progress bar.
    """

    def __init__(self, model, x_train, y_train, factr=1e5, m=50, maxls=50, maxiter=15000):
        """
        Args:
            model: optimization target model.
            samples: training samples.
            factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
                   1e7 for moderate accuracy; 10.0 for extremely high accuracy.
            m: maximum number of variable metric corrections used to define the limited memory matrix.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
        """

        # set attributes
        self.model = model
        self.x_train = x_train#[ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = y_train#[ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.loss4 = []
        self.factr = factr
        self.ModelInfo = benchmark()
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.losses = []
        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params( {
            'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})

    def set_weights(self, flat_weights):
        """
        Set weights to the model.

        Args:
            flat_weights: flatten weights.
        """

        # get model weights
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate loss and gradients for weights as tf.Tensor.

        Args:
            x: input data.

        Returns:
            loss and gradients for weights as tf.Tensor.
        """
        
        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
            loss1 = tf.reduce_mean(tf.keras.losses.mse(self.model(x)[0], y[0]))
            loss2 = tf.reduce_mean(tf.keras.losses.mse(self.model(x)[1], y[1]))
            
            loss_array = [loss1, loss2]
            
            if self.ModelInfo.mode == "PINNs":
                loss3 = tf.reduce_mean(tf.keras.losses.mse(self.model(x)[2], y[2]))
                loss4 = tf.reduce_mean(tf.keras.losses.mse(self.model(x)[3], y[3]))
                
                loss_array = loss_array + [loss3, loss4]
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads, loss_array

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.

        Args:
            weights: flatten weights.

        Returns:
            loss and gradients for weights as ndarray.
        """

        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads, loss_array = self.tf_evaluate(self.x_train, self.y_train)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads, loss_array

    def callback(self, weights):
        """
        Callback that prints the progress to stdout.

        Args:
            weights: flatten weights.
        """
        self.progbar.on_batch_begin(0)
        loss, _, loss_array = self.evaluate(weights)
        
        self.losses.append(loss)
        self.loss1.append(loss_array[0])
        self.loss2.append(loss_array[1])
        if self.ModelInfo.mode == 'PINNs':
            self.loss3.append(loss_array[2])
            self.loss4.append(loss_array[3])
        
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """

        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ])
        # optimize the weight vector
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
            factr=self.factr, m=self.m, maxls=self.maxls, maxiter=self.maxiter,
            callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()
        
        
        ModelInfo = benchmark()
        
        plt.title('Losses')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        
        
        plt.plot(np.log(self.loss1), label = 'Loss '+ self.ModelInfo.mode)
        plt.plot(np.log(self.loss2), label = 'Loss u_ini')
        
        if self.ModelInfo.mode == 'PINNs':
            plt.plot(np.log(self.loss3), label = 'Loss du_ini')
            plt.plot(np.log(self.loss4), label = 'Loss u_bnd')
            
        plt.plot(np.log(self.losses), label = 'Overall loss', color = 'black')
        plt.legend()
        plt.savefig('losses/combined_losses_reduction_'+ ModelInfo.mode +'.png')
        plt.show()
        
        loss1 = np.array(self.loss1)
        np.savetxt('losses/CSV/loss_PDE_'+ ModelInfo.mode +'.csv', loss1, delimiter = ',')
        loss2 = np.array(self.loss2)
        np.savetxt('losses/CSV/loss_u_ini_'+ ModelInfo.mode +'.csv', loss2, delimiter = ',')
        
        if self.ModelInfo.mode == 'PINNs':
            loss3 = np.array(self.loss3)
            np.savetxt('losses/CSV/loss_du_ini_'+ ModelInfo.mode +'.csv', loss3, delimiter = ',')
            loss4 = np.array(self.loss4)
            np.savetxt('losses/CSV/loss_u_bnd_'+ ModelInfo.mode +'.csv', loss4, delimiter = ',')
            
            
        plt.title('Loss in L_BFGS')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.plot(np.arange(len(self.losses)), np.log(self.losses))
        act1 = np.array(self.losses)
        np.savetxt('losses/CSV/tanh_'+ ModelInfo.mode +'.csv', act1, delimiter = ',')
        plt.savefig('losses/Loss_Wave_'+ ModelInfo.mode +'.png')
        plt.show()
        
        
