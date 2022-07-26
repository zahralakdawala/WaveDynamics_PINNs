import sys
sys.path.append('../SWE')

import tensorflow as tf
from lib.layer import *

from benchmark import *

ModelInfo = InputVariables()


class PINN:
    """
    Build a physics informed neural network (PINN) model for the steady Navier-Stokes equation.

    Attributes:
        network: keras network model with input (x, y) and output (psi, p).
        rho: density.
        nu: viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, g = 9.8):
        """
        Args:
            network: keras network model with input (x, y) and output (psi, p).
            rho: density.
            nu: viscosity.
        """
        ModelInfo = InputVariables()
        self.network = network
        self.g = g
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the steady Navier-Stokes equation.

        Returns:
            PINN model for the steady Navier-Stokes equation with
                input: [ (x, y) relative to equation,
                         (x, y) relative to boundary condition ],
                output: [ (u, v) relative to equation (must be zero),
                          (psi, psi) relative to boundary condition (psi is duplicated because outputs require the same dimensions),
                          (u, v) relative to boundary condition ]
        """

        # equation input: (x, y)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # boundary condition
        tx_bnd = tf.keras.layers.Input(shape=(2,))
        # boundary condition
        tx_ini = tf.keras.layers.Input(shape=(2,))
        # Numerical solution
        tx_sol = tf.keras.layers.Input(shape=(2,))

        # Solution learning
        vy_sol = self.network(tx_sol)

        if ModelInfo.mode in ['PINNs', 'dataAndPhysics']:
            # compute gradients relative to equation
            #v, y, dvdt, dvdx, dydt, dydx = self.grads(tx_eqn)
        
            #m  = ModelInfo.m
            #Sb = ModelInfo.Sb
            
            # compute gradients relative to equation
            h, u, dhdt, dhdx, dudt, dudx = self.grads(tx_eqn)
    
            # compute equation loss
    
            duhdt = h*dudt + u*dhdt
            duhdx = h*dudx + u*dhdx
            '''
            Sf = lambda u, h : (m**2)*u*abs(u) / ((h)**(4/3))
            
            eqn1 = dudt + u*dudx + self.g*dhdx + self.g*(Sb - Sf(u, h))
            eqn2 = dhdt + duhdx 
            '''
            eqn1 = dhdt + duhdx 
            eqn2 = duhdt +2*h*u*(dudx) + (u**2)*dhdx + self.g*h*dhdx
            
            #vy_eqn = tf.concat([v_eqn, y_eqn], axis=-1)
            vy_eqn = tf.concat([eqn1, eqn2], axis=-1)

            # Initial condition
            vy_ini = self.network(tx_ini)
          
            # compute gradients relative to boundary condition
            
            h_bnd, u_bnd, _, dh_bnd, _, du_bnd = self.grads(tx_bnd)
            # compute boundary condition loss
            duhbnd = h*du_bnd + u*dh_bnd
            
            bnd = tf.concat([dh_bnd, dh_bnd], axis=-1)
            
            if ModelInfo.mode == "dataAndPhysics":
                return tf.keras.models.Model(
                    inputs=[tx_eqn, tx_ini, tx_sol], outputs=[vy_eqn, vy_ini, vy_sol])
            
            else:
                return tf.keras.models.Model(
                    inputs=[tx_eqn, tx_ini], outputs=[vy_eqn, vy_ini])
        
        elif ModelInfo.mode == "data":
            return tf.keras.models.Model(
                    inputs=[tx_sol], outputs=[vy_sol])