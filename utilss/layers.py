# layers.py
#
# Author: Jie Zhu
# Tested with Python version 3.8 and TensorFlow 2.0

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Masking, Layer
from tensorflow.keras import initializers

__all__ = ['ExternalMasking']



class ExternalMasking(Masking):
    """An extension of `Masking` layer.
    Use the second input to determine the masking of the first input.
    """
    def compute_mask(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('Inputs to ExternalMasking should be a list of 2 tensors.')
        return super(ExternalMasking, self).compute_mask(inputs[-1])

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('Inputs to ExternalMasking should be a list of 2 tensors.')
        boolean_mask = K.any(K.not_equal(inputs[-1], self.mask_value),axis=-1, keepdims=True)
        return inputs[0] * K.cast(boolean_mask, K.dtype(inputs[0]))

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('input_shape to ExternalMasking should be a list of 2 tensors.')
        if input_shape[0][:2] != input_shape[1][:2]:
            raise ValueError('The first two dimensions of the two inputs should be the '
                             'same, but got {} and {} from them.'.format(
                input_shape[0][:2], input_shape[1][:2])
            )
        return input_shape[0]

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def _get_layers_scope_dict(self):
        return {
            'ExternalMasking': ExternalMasking,
        }



class concateDim(Layer):
    def __init__(self, output_dim,init = '', **kwargs):
        self.output_dim = output_dim
        self.init = init
        super(concateDim, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(concateDim, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
            return  mask

    def call(self, inputs, mask=None):
        self.out = K.sigmoid(K.mean(K.dot(inputs, self.kernel), axis=1))
        return self.out

    def compute_output_shape(self, input_shape):
        return (K.int_shape(self.out))

    def get_config(self):
        config = {'output_dim': self.output_dim}
        return dict(list(config.items()))


class reducedim(Layer):
    def __init__(self, output_dim,init = '', **kwargs):
        self.output_dim = output_dim
        self.init = init
        super(reducedim, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer= initializers.ones,
                                      trainable=False)
        super(reducedim, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
            return  mask

    def call(self, inputs, mask=None):
        self.out = K.mean(inputs, axis=1)
        return self.out

    def compute_output_shape(self, input_shape):
        return (K.int_shape(self.out))

    def get_config(self):
        config = {'output_dim': self.output_dim}
        return dict(list(config.items()))
