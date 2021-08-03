import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Conv2D


class PConv2D(Conv2D):
    def __init__(self, filters, kernel_size, padding='same', **kwargs):
        """Overwrite the following:
            * Default to zero-padding
            * Reformulate to two inputs instead of only 1
        """
        super().__init__(filters, kernel_size, padding=padding, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """Adapted from original Conv2D layer of Keras
        param input_shape: list of dimensions for [img, mask]
        """
        # Run the usual Conv2D build() method & reenforce input spec
        super().build(input_shape=input_shape[0])
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        # Mask kernel
        self.kernel_mask = K.constant(1, shape=self.kernel.shape)
        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        '''
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        '''

        # Apply convolutions to mask
        mask_output = K.conv2d(
            inputs[1], self.kernel_mask,
            strides=self.strides,
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        # Run original convolution operation
        img_output = self._convolution_op(inputs[0]*inputs[1], self.kernel)

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(img_output, self.bias, data_format=self.data_format)

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]
