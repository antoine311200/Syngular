import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Optimizer

import tensornetwork as tn
import numpy as np


class TensorGradientDescent(Optimizer):

    def __init__(self, learning_rate=0.01, name="Tensor Gradient Descent Optimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._is_first = True

    def _create_slots(self, var_list):
        # Adding previous variables
        for var in var_list:
            self.add_slot(var, "pv")
        # Adding previous gradients
        for var in var_list:
            self.add_slot(var, "pg")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        
        new_var_m = var - grad * lr_t

        # Extract the previous values of Variables and Gradients
        pv_var = self.get_slot(var, "pv")
        pg_var = self.get_slot(var, "pg")
        
        # If it first time, use just the traditional method
        if self._is_first:
            self._is_first = False
            new_var = new_var_m
        else:
	        # create a boolean tensor contain true and false
            # True will be where the gradient haven't changed the sign and False will be the case where the gradients have changed sign
            cond = grad*pg_var >= 0
	
	        # Compute the average of previous weight and current. Though we will be using only few of these. 
            #Of course, it is prone to overflow. We can also compute the avg using a + (b -a)/2.0
            avg_weights = (pv_var + var)/2.0
	 
            # tf.where picks the value from new_var_m where the cond is True otherwise it takes from avg_weights
            # We must avoid the for loops
            new_var = tf.where(cond, new_var_m, avg_weights)
        
        # Finally we are saving current values in the slots.
        pv_var.assign(var)
        pg_var.assign(grad)
    
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }