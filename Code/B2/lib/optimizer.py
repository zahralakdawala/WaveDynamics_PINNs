# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:05:09 2022

@author: Admin
"""
import tensorflow as tf
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../SWE')
from benchmark import *

ModelInfo = InputVariables()

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.

    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: function convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10.0 for extremely high accuracy.
        pgtol: gradient convergence condition.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: log metrics
        progbar: progress bar
    """

    def __init__(self, model, x_train, y_train, factr=10, pgtol=1e-12, m=500, maxls=500, maxiter=10000):
        """
        Args:
            model: optimization target model.
            samples: training samples.
            factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
                   1e7 for moderate accuracy; 10.0 for extremely high accuracy.
            pgtol: gradient convergence condition.
            m: maximum number of variable metric corrections used to define the limited memory matrix.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
        """

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.pgtol = pgtol
        self.losses = []
        self.loss1 = []
        
        if (ModelInfo.mode == 'PINNs') or (ModelInfo.mode == 'dataAndPhysics'):
            self.loss2 = []
        if ModelInfo.mode == 'dataAndPhysics':
            self.loss3 = []
            
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
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
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.model(x), y))
            loss1 = tf.reduce_mean(tf.keras.losses.mse(self.model(x)[0], y[0]))
            loss_arrays = [loss1]
            if (ModelInfo.mode == 'PINNs') or (ModelInfo.mode == 'dataAndPhysics'):
                loss2 = tf.reduce_mean(tf.keras.losses.mse(self.model(x)[1], y[1]))
                loss_arrays = loss_arrays + [loss2]
            if ModelInfo.mode == 'dataAndPhysics':
                loss3 = tf.reduce_mean(tf.keras.losses.mse(self.model(x)[2], y[2]))
                loss_arrays = loss_arrays + [loss3]
                
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads, loss_arrays

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
        loss, grads, loss_arrays = self.tf_evaluate(self.x_train, self.y_train)
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads, loss_arrays

    def callback(self, weights):
        """
        Callback that prints the progress to stdout.

        Args:
            weights: flatten weights.
        """
        self.progbar.on_batch_begin(0)
        loss, _, loss_arrays = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))
        self.losses.append(loss)
        
        self.loss1.append(loss_arrays[0])
        
        if (ModelInfo.mode == 'PINNs') or (ModelInfo.mode == 'dataAndPhysics'):
            self.loss2.append(loss_arrays[1])
        if ModelInfo.mode == 'dataAndPhysics':
            self.loss3.append(loss_arrays[2])

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
            factr=self.factr, pgtol=self.pgtol, m=self.m,
            maxls=self.maxls, maxiter=self.maxiter, callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()
        
        plt.title('Losses')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        
        if ModelInfo.mode == 'data':
            plt.plot(np.log(self.loss1), label = 'Loss data')
        if ModelInfo.mode == 'PINNs':
            plt.plot(np.log(self.loss1), label = 'Loss PDE')
            plt.plot(np.log(self.loss2), label = 'Loss ini')
        if ModelInfo.mode == 'dataAndPhysics':
            plt.plot(np.log(self.loss1), label = 'Loss PDE')
            plt.plot(np.log(self.loss2), label = 'Loss ini')      
            plt.plot(np.log(self.loss3), label = 'Loss data')

        plt.plot(np.log(self.losses), label = 'Overall loss', color = 'black')
        plt.legend()
        plt.savefig('losses/combined_loss_bc' + str(ModelInfo.benchmark) + '_' + ModelInfo.mode+'.png')
        plt.show()
        
        plt.title('Loss in L_BFGS')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.plot(np.arange(len(self.losses)), np.log(self.losses))
        plt.savefig('losses/loss_NS_'+ModelInfo.mode+'.png')
        plt.show()