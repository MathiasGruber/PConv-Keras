from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation, Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from libs.pconv_layer import PConv2D
from libs.gconv_layer import GConv2D
from libs.losses import l1_error, gram_matrix, total_variation


class UNetModel(Model):
    """
    Overwrite the training step to inplement the proper loss function
    See: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """

    def __init__(self,
        img_rows=512, img_cols=512,
        l1_factor=1.0, l2_factor=1.0, l3_factor=0.03,
        l4_factor=0.0003, l5_factor=0.0005, l6_factor=1.0,
        *args, **kwargs
    ):
        """Define the losses to track"""
        super(UNetModel, self).__init__(*args, **kwargs)

        # Image dimensions
        self.img_rows = img_rows
        self.img_cols = img_cols

        # VGG network for perceptual & style losses
        self.vgg = self.build_vgg(img_rows, img_cols)

        # Loss adjustment factors (for optionally modifying losses)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.l3_factor = l3_factor
        self.l4_factor = l4_factor
        self.l5_factor = l5_factor
        self.l6_factor = l6_factor

        # Losses
        self.l1 = tf.keras.metrics.Mean(name="l1")
        self.l2 = tf.keras.metrics.Mean(name="l2")
        self.l3 = tf.keras.metrics.Mean(name="l3")
        self.l4 = tf.keras.metrics.Mean(name="l4")
        self.l5 = tf.keras.metrics.Mean(name="l5")
        self.l6 = tf.keras.metrics.Mean(name="l6")
        self.total_loss = tf.keras.metrics.Mean(name='loss')

        # Metrics
        self.metric_l1 = tf.keras.metrics.MeanAbsoluteError(name="L1_error")
        self.metric_psnr = tf.keras.metrics.Mean(name="PSNR")
        self.metric_ssim = tf.keras.metrics.Mean(name="SSIM")
        self.metric_maskratio = tf.keras.metrics.Mean(name="Mask_Ratio")

    def build_vgg(self, img_rows, img_cols):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """

        # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
        vgg_layers = [3, 6, 10]

        # Input image to extract features from
        img = Input(shape=(img_rows, img_cols, 3))

        # Input is between 0 and 1, rescale to 255
        processed = Lambda(lambda x: x * 255)(img)

        # Preprocess input
        processed = preprocess_input(processed)

        # Get the vgg network from Keras applications
        vgg = VGG16(weights='imagenet', include_top=False)

        # Create a VGG network from appropriate output layers
        vgg = Model(inputs=vgg.input, outputs=[vgg.layers[i].output for i in vgg_layers])

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(processed))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model

    def test_step(self, data):

        # Inputs are image and mask, output is image
        (masked, mask), y_true = data

        # Forward pass
        y_pred = self((masked, mask), training=False)

        # Track the PSNR, SSIM and L1 metrics
        self.metric_l1.update_state(y_true, y_pred)
        self.metric_psnr.update_state(tf.image.psnr(y_true, y_pred, max_val=1))
        self.metric_ssim.update_state(tf.image.ssim(y_true, y_pred, max_val=1))

        # Return all metrics
        return {m.name: m.result() for m in [self.metric_psnr, self.metric_l1, self.metric_ssim]}

    def train_step(self, data):

        # Inputs are image and mask, output is image
        (masked, mask), y_true = data

        # Cast the mask as float32
        mask = tf.cast(mask, tf.float32)

        with tf.GradientTape() as tape:

            # Forward pass
            y_pred = self((masked, mask), training=True)

            # Fill in hole pixels with predictions in original image.
            y_comp = mask * y_true + (1-mask) * y_pred

            # Compute VGG outputs
            vgg_out = self.vgg(y_pred)
            vgg_gt = self.vgg(y_true)
            vgg_comp = self.vgg(y_comp)

            # Loss 1: mean absolyte error outside hole region
            l1 = l1_error(mask * y_true, mask * y_pred)
            l1 *= self.l1_factor

            # Loss 2: mean absolute error inside hole region
            l2 = 6 * l1_error((1-mask) * y_true, (1-mask) * y_pred)
            l2 *= self.l2_factor

            # Loss 3: Perceptual loss based on VGG features
            # Scaled with 0.03 at end to make size comparable to paper
            l3 = tf.zeros_like(l1)
            for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
                l3 += 0.05 * (l1_error(o, g) + l1_error(c, g)) * self.l3_factor

            # Loss 4: Style loss on raw output
            # Scaled with 0.0003 at end to make size comparable to paper
            l4 = tf.zeros_like(l1)
            for o, g in zip(vgg_out, vgg_gt):
                l4 += 120 * l1_error(gram_matrix(o), gram_matrix(g)) * self.l4_factor

            # Loss 5: Style loss on computed output
            # Scaled with 0.0005 at end to make size comparable to paper
            l5 = tf.zeros_like(l1)
            for o, g in zip(vgg_comp, vgg_gt):
                l5 += 120 * l1_error(gram_matrix(o), gram_matrix(g)) * self.l5_factor

            # Loss 6: Total variation loss for smoothing edges
            l6 = 0.1 * total_variation(y_comp)
            l6 *= self.l6_factor

            # Calculate total loss as per equation in paper
            total_loss = l1 + l2 + l3 + l4 + l5 + l6

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our losses
        self.l1.update_state(l1)
        self.l2.update_state(l2)
        self.l3.update_state(l3)
        self.l4.update_state(l4)
        self.l5.update_state(l5)
        self.l6.update_state(l6)
        self.total_loss.update_state(total_loss)

        # Compute the metrics
        self.metric_l1.update_state(y_true, y_pred)
        self.metric_psnr.update_state(tf.image.psnr(y_true, y_pred, max_val=1))
        self.metric_ssim.update_state(tf.image.ssim(y_true, y_pred, max_val=1))
        self.metric_maskratio.update_state(tf.math.reduce_mean(tf.cast(1-mask, tf.float32), axis=[1,2,3]))

        # Return all metrics
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.total_loss,
            self.metric_l1, self.metric_psnr, self.metric_ssim, self.metric_maskratio
        ]



class InpaintingUnet(object):

    def __init__(
        self,
        img_rows=512, img_cols=512,
        train_bn=True, lr=0.0002,
        load_weights=None,
        net_name='default',
        conv_layer='pconv'
    ):
        """Create the UNet for inpainting based on paper: https://arxiv.org/abs/1804.07723

        Args:
            img_rows    : image height. Set to None for variable size
            img_cols    : image width. Set to None for variable size
            train_bn    : whether to train batch normalization layers
            load_weights: Tensorflor SavedModel path to load weights
            net_name    : Name of this network (used in logging).
            conv_layer  : Convolution layer (pconv|gconv)
        """

        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_overlap = 30
        self.net_name = net_name
        self.conv_layer = conv_layer

        # Assertions
        assert img_rows >= 256, 'Height must be >256 pixels'
        assert img_cols >= 256, 'Width must be >256 pixels'
        assert conv_layer in ['pconv', 'gconv'], f'Unknown convolutional layer {conv_layer}'

        # Get number of available GPUs
        self.gpus = len(tf.config.list_physical_devices('GPU'))

        # Build and compile model
        self.model = self.build_unet(train_bn=train_bn)

        # Load weights into model
        if load_weights is not None:
            self.model.load_weights(load_weights)

        # Compile the model
        self.compile_pconv_unet(self.model, lr=lr)

    def build_unet(self, train_bn=True):

        def _create():

            # INPUTS
            inputs_img = Input((self.img_rows, self.img_cols, 3), name='inputs_img')
            inputs_mask = Input((self.img_rows, self.img_cols, 3), name='inputs_mask')

            # CONVOLUTION LAYER
            # PConv2D: Partial Convolution (https://arxiv.org/abs/1804.07723)
            # GConv2D: Gated Convolution (https://arxiv.org/abs/1806.03589)
            MaskConvLayer = PConv2D if self.conv_layer == 'pconv' else GConv2D

            # ENCODER
            def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
                conv, mask = MaskConvLayer(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
                if bn:
                    conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
                conv = Activation('relu')(conv)
                encoder_layer.counter += 1
                return conv, mask
            encoder_layer.counter = 0

            e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
            e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
            e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
            e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
            e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
            e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
            e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3)
            e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 512, 3)

            # DECODER
            def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
                up_img = UpSampling2D(size=(2,2))(img_in)
                up_mask = UpSampling2D(size=(2,2))(mask_in)
                concat_img = Concatenate(axis=3)([e_conv,up_img])
                concat_mask = Concatenate(axis=3)([e_mask,up_mask])
                conv, mask = MaskConvLayer(filters, kernel_size, padding='same')([concat_img, concat_mask])
                if bn:
                    conv = BatchNormalization()(conv)
                conv = LeakyReLU(alpha=0.2)(conv)
                return conv, mask

            d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 512, 3)
            d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 512, 3)
            d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3)
            d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3)
            d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
            d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
            d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
            d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)
            outputs = Conv2D(3, 1, activation = 'sigmoid', name='outputs_img')(d_conv16)

            # Setup the model & return it
            model = UNetModel(
                inputs=[inputs_img, inputs_mask],
                outputs=outputs,
                img_rows=self.img_rows,
                img_cols=self.img_cols
            )
            return model

        # Create UNet-like model
        print(f"Running model on {self.gpus} GPUs")
        if self.gpus <= 1:
            model = _create()
        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = _create()

        return model

    def compile_pconv_unet(self, model, lr=0.0002):
        """Compile the model and enforce the given learning rate."""
        model.compile('adam')
        model.optimizer.learning_rate.assign(lr)

    def fit(self, generator, *args, **kwargs):
        """Fit the U-Net to a [(masked_images, masks), target_images] generator"""
        self.model.fit(generator, *args, **kwargs)

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')