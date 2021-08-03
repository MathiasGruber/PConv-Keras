import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Conv2D


class GConv2D(Conv2D):
    """Layer for gated convolution, see paper: https://arxiv.org/pdf/1806.03589.pdf"""

    def __init__(self, filters, kernel_size, padding='same', **kwargs):
        """Overwrite the following:
            * Default to zero-padding
            * Double the filters, since we need to learn the mask update as well
            * Reformulate to two inputs instead of only 1
        """
        super().__init__(filters*2, kernel_size, padding=padding, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """Run the usual Conv2D build() method & reenforce input spec"""
        # Get channel axis & deconstruct input shape
        img_shape, mask_shape = input_shape
        # Create a concatenate shape tensor
        concat_shape = tensor_shape.TensorShape([
            img_shape[0], img_shape[1], img_shape[2], img_shape[3] + mask_shape[3]
        ])
        # Build convolutional layer with concatenate shape
        super().build(input_shape=concat_shape)
        # Re-enforce input spec
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        '''
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        '''

        # Concatenate inputs with the mask
        inputs = K.concatenate([inputs[0], inputs[1]], axis=-1)

        # Run original convolution operation
        img_output = self._convolution_op(inputs, self.kernel)

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(img_output, self.bias, data_format=self.data_format)

        # Split out into image and mask again
        img_output, mask_output = tf.split(img_output, num_or_size_splits=2, axis=self._get_channel_axis())

        # Apply activation on the image and sigmoid on mask
        if self.activation is not None:
            img_output = self.activation(img_output)
        mask_output = K.sigmoid(mask_output)

        # Multiply the mask with the image (See paper equation)
        img_output = img_output * mask_output

        # Return both newimage and mask
        return [img_output, mask_output]
