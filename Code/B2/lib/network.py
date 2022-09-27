# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:03:16 2022

@author: Admin
"""

import tensorflow as tf

class Network:
    """
    Build a physics informed neural network (PINN) model for the steady Navier-Stokes equations.
    Attributes:
        activations: custom activation functions.
    """

    def __init__(self):
        """
        Setup custom activation functions.
        """
        self.activations = {
            'sigmoid': 'sigmoid',
            'relu' : 'relu',
            'tanh' : 'tanh'
        }


    def build(self, num_inputs=2, layers=[20, 20, 20, 20, 20], activation= 'tanh', num_outputs=2):
        """
        Build a PINN model for the steady Navier-Stokes equation with input shape (x,y) and output shape (psi, p).
        Args:
            num_inputs: number of input variables. Default is 2 for (x, y).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 2 for (psi, p).
        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=self.activations[activation],
                kernel_initializer='orthogonal')(x)

        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='orthogonal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)