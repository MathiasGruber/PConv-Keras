import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from libs.pconv_layer import PConv2D
from libs.losses import l1_error, gram_matrix, total_variation


class PConvModel(Model):
    """
    Overwrite the training step to inplement the proper loss function
    See: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """

    def __init__(self, vgg_weights='imagenet', img_rows=512, img_cols=512, *args, **kwargs):
        """Define the losses to track"""
        super(PConvModel, self).__init__(*args, **kwargs)

        # Image dimensions
        self.img_rows = img_rows
        self.img_cols = img_cols

        # VGG network for perceptual & style losses
        self.vgg = self.build_vgg(vgg_weights, img_rows, img_cols)

        # Losses
        self.l1 = tf.keras.metrics.Mean(name="l1")
        self.l2 = tf.keras.metrics.Mean(name="l2")
        self.l3 = tf.keras.metrics.Mean(name="l3")
        self.l4 = tf.keras.metrics.Mean(name="l4")
        self.l5 = tf.keras.metrics.Mean(name="l5")
        self.l6 = tf.keras.metrics.Mean(name="l6")
        self.total_loss = tf.keras.metrics.Mean(name='loss')

        # Metrics
        self.metric_l1 = tf.keras.metrics.MeanAbsoluteError(name="L1 error")
        self.metric_psnr = tf.keras.metrics.Mean(name="PSNR")
        self.metric_ssim = tf.keras.metrics.Mean(name="SSIM")
        self.metric_maskratio = tf.keras.metrics.Mean(name="Mask Ratio")

    def build_vgg(self, weights, img_rows, img_cols):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """

        # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
        vgg_layers = [3, 6, 10]

        # Scaling for VGG input
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Input image to extract features from
        img = Input(shape=(img_rows, img_cols, 3))

        # Mean center and rescale by variance as in PyTorch
        processed = Lambda(lambda x: (x-mean) / std)(img)

        # Get the vgg network from Keras applications
        if weights in ['imagenet', None]:
            vgg = VGG16(weights=weights, include_top=False)
        else:
            vgg = VGG16(weights=None, include_top=False)
            vgg.load_weights(weights, by_name=True)

        # Create a VGG network from appropriate output layers
        vgg = Model(inputs=vgg.input, outputs=[vgg.layers[i].output for i in vgg_layers])

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(processed))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model

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

            # Loss 2: mean absolute error inside hole region
            l2 = 6 * l1_error((1-mask) * y_true, (1-mask) * y_pred)

            # Loss 3: Perceptual loss based on VGG features
            l3 = tf.zeros_like(l1)
            for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
                l3 += 0.05 * (l1_error(o, g) + l1_error(c, g))

            # Loss 4: Style loss on raw output
            l4 = tf.zeros_like(l1)
            for o, g in zip(vgg_out, vgg_gt):
                l4 += 120 * l1_error(gram_matrix(o), gram_matrix(g))

            # Loss 5: Style loss on computed output
            l5 = tf.zeros_like(l1)
            for o, g in zip(vgg_comp, vgg_gt):
                l5 += 120 * l1_error(gram_matrix(o), gram_matrix(g))

            # Loss 6: Total variation loss for smoothing edges
            l6 = 0.1 * total_variation(y_comp)

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

        return {
            "loss": self.total_loss.result(),
            "l1": self.l1.result(),
            "l2": self.l2.result(),
            "l3": self.l3.result(),
            "l4": self.l4.result(),
            "l5": self.l5.result(),
            "l6": self.l6.result(),
            "metric_L1": self.metric_l1.result(),
            "metric_PSNR": self.metric_psnr.result(),
            "metric_SSIM": self.metric_ssim.result(),
            "metric_MaskRatio": self.metric_maskratio.result(),
        }

    @property
    def metrics(self):
        return [
            self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.total_loss,
            self.metric_l1, self.metric_psnr, self.metric_ssim, self.metric_maskratio
        ]



class PConvUnet(object):

    def __init__(
        self,
        img_rows=512, img_cols=512,
        train_bn=True, lr=0.0002,
        load_weights=None,
        vgg_weights="imagenet",
        inference_only=False,
        net_name='default'
    ):
        """Create the PConvUnet. If variable image size, set img_rows and img_cols to None

        Args:
            img_rows (int): image height.
            img_cols (int): image width.
            vgg_weights (str): which weights to pass to the vgg network.
            inference_only (bool): initialize BN layers for inference.
            net_name (str): Name of this network (used in logging).
        """

        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.vgg_weights = vgg_weights
        self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name

        # Assertions
        assert self.img_rows >= 256, 'Height must be >256 pixels'
        assert self.img_cols >= 256, 'Width must be >256 pixels'

        # Set current epoch
        self.current_epoch = 0

        # Get number of available GPUs
        self.gpus = len(tf.config.list_physical_devices('GPU'))

        # Build and compile model
        self.model = self.build_pconv_unet(train_bn=train_bn)

        # Load weights into model
        if load_weights is not None:
            epoch = int(os.path.basename(load_weights).split('.')[1].split('-')[0])
            assert epoch > 0, "Could not parse weight file. Should include the epoch"
            self.current_epoch = epoch
            self.model.load_weights(load_weights)

        # Compile the model
        self.compile_pconv_unet(self.model, lr=lr)

    def build_pconv_unet(self, train_bn=True):

        def _create():
            # INPUTS
            inputs_img = Input((self.img_rows, self.img_cols, 3), name='inputs_img')
            inputs_mask = Input((self.img_rows, self.img_cols, 3), name='inputs_mask')

            # ENCODER
            def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
                conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
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
                conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
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
            model = PConvModel(
                inputs=[inputs_img, inputs_mask],
                outputs=outputs,
                vgg_weights=self.vgg_weights,
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
        model.compile(optimizer = Adam(lr=lr))

    def fit(self, generator, *args, **kwargs):
        """Fit the U-Net to a [(masked_images, masks), target_images] generator"""
        self.model.fit(generator, *args, **kwargs)

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    @staticmethod
    def PSNR(y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
        two values (4.75) as MAX_I
        """
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Prediction functions
    ######################
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
