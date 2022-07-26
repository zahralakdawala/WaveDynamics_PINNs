import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute derivatives for the steady Navier-Stokes equation.
    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        self.g = 9.8
        super().__init__(**kwargs)

    def call(self, tx):
        """
        Computing derivatives for the steady Navier-Stokes equation.
        Args:
            xy: input variable.
        Returns:
            psi: stream function.
            p_grads: pressure and its gradients.
            u_grads: u and its gradients.
            v_grads: v and its gradients.
        """
        
        with tf.GradientTape(persistent=True) as ggg:
            ggg.watch(tx)
            vy   = self.model(tx)
            v    = vy[..., 0, tf.newaxis]
            y    = vy[..., 1, tf.newaxis]
            
            
        dvdtx    = ggg.batch_jacobian(v, tx)
        dvdt     = dvdtx[..., 0]
        dvdx     = dvdtx[..., 1]
    
        dydtx   = ggg.batch_jacobian(y, tx)
        dydt    = dydtx[..., 0]
        dydx    = dydtx[..., 1]
        
        del ggg
        
        return v, y, dvdt, dvdx, dydt, dydx 