# (c) 2019 Joakim Berntsson
# Multi layer perception module.
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout


class FeedForwardNetwork(tf.keras.Model):
    """Feed forward network with support for:
    * arbitrary number of layers
    * batch normalization
    * l2 weight regularizer
    * dropout.
    """

    def __init__(self, units: list, l2_weight_decay: float = None, 
                 use_bn: bool = False, dropout: float = None,
                 use_sigmoid: bool = True):
        super(FeedForwardNetwork, self).__init__()

        self._net = []

        for i, unit in enumerate(units):
            reg = None
            if l2_weight_decay:
                reg = tf.contrib.layers.l2_regularizer(scale=l2_weight_decay)
            fc = Dense(unit, kernel_regularizer=reg, name="fc{}".format(i))
            self._net.append(fc)
            if use_bn:
                bn = BatchNormalization(name="bnorm{}".format(i))
                self._net.append(bn)
            if dropout:
                do = Dropout(dropout, name="dropout{}".format(i))
                self._net.append(do)
            self._net.append(tf.keras.activations.relu)
        
        regf = None
        if l2_weight_decay:
            regf = tf.contrib.layers.l2_regularizer(scale=l2_weight_decay)
        final = Dense(1, kernel_regularizer=regf, name="final")
        self._net.append(final)
        if use_sigmoid:
            self._net.append(tf.keras.activations.sigmoid)

    @property
    def trainable_variables(self):
        variables = []
        for comp in self._net:
            if hasattr(comp, "trainable_weights"):
                variables += comp.trainable_weights
        return variables

    def call(self, inputs, training=True):
        """Forward propagation for the network."""
        with tf.variable_scope("simple_network"):
            intermed = inputs
            for comp in self._net:
                if isinstance(comp, BatchNormalization):
                    intermed = comp(intermed, training=training)
                else:
                    intermed = comp(intermed)
        return intermed

    def regularize_loss(self):
        losses = []
        for comp in self._net:
            if hasattr(comp, "losses"):
                losses += comp.losses
        return tf.add_n(losses) if len(losses) > 0 else 0
