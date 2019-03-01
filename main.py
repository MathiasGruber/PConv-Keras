import os
import gc
import datetime
import numpy as np
import pandas as pd
import cv2

from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras import backend as K
from keras.utils import Sequence
from keras_tqdm import TQDMCallback

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from libs.pconv_model import PConvUnet
from libs.util import MaskGenerator


# Sample call
r"""
# Train on CelebaHQ
python main.py --name CelebHQ --train C:\Documents\Kaggle\celebaHQ-512\train\ --validation C:\Documents\Kaggle\celebaHQ-512\val\ --test C:\Documents\Kaggle\celebaHQ-512\test\ --checkpoint "C:\Users\Mathias Felix Gruber\Documents\GitHub\PConv-Keras\data\logs\imagenet_phase1_paperMasks\weights.35-0.70.h5"
"""


def parse_args():
    parser = ArgumentParser(description='Training script for PConv inpainting')

    parser.add_argument(
        '-stage', '--stage',
        type=str, default='train',
        help='Which stage of training to run',
        choices=['train', 'finetune']
    )

    parser.add_argument(
        '-train', '--train',
        type=str,
        help='Folder with training images'
    )
    
    parser.add_argument(
        '-validation', '--validation',
        type=str,
        help='Folder with validation images'
    )

    parser.add_argument(
        '-test', '--test',
        type=str,
        help='Folder with testing images'
    )
        
    parser.add_argument(
        '-name', '--name',
        type=str, default='myDataset',
        help='Dataset name, e.g. \'imagenet\''
    )
        
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=4,
        help='What batch-size should we use'
    )

    parser.add_argument(
        '-test_path', '--test_path',
        type=str, default='./data/test_samples/',
        help='Where to output test images during training'
    )
        
    parser.add_argument(
        '-weight_path', '--weight_path',
        type=str, default='./data/logs/',
        help='Where to output weights during training'
    )
        
    parser.add_argument(
        '-log_path', '--log_path',
        type=str, default='./data/logs/',
        help='Where to output tensorboard logs during training'
    )

    parser.add_argument(
        '-vgg_path', '--vgg_path',
        type=str, default='./data/logs/pytorch_to_keras_vgg16.h5',
        help='VGG16 weights trained on PyTorch with pixel scaling 1/255.'
    )

    parser.add_argument(
        '-checkpoint', '--checkpoint',
        type=str, 
        help='Previous weights to be loaded onto model'
    )
        
    return  parser.parse_args()


class AugmentingDataGenerator(ImageDataGenerator):
    """Wrapper for ImageDataGenerator to return mask & image"""
    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)        
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample            
            mask = np.stack([
                mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    if args.stage == 'finetune' and not args.checkpoint:
        raise AttributeError('If you are finetuning your model, you must supply a checkpoint file')

    # Create training generator
    train_datagen = AugmentingDataGenerator(  
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        args.train, 
        MaskGenerator(512, 512, 3),
        target_size=(512, 512), 
        batch_size=args.batch_size
    )

    # Create validation generator
    val_datagen = AugmentingDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        args.validation, 
        MaskGenerator(512, 512, 3), 
        target_size=(512, 512), 
        batch_size=args.batch_size, 
        classes=['val'], 
        seed=42
    )

    # Create testing generator
    test_datagen = AugmentingDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        args.test, 
        MaskGenerator(512, 512, 3), 
        target_size=(512, 512), 
        batch_size=args.batch_size, 
        seed=42
    )

    # Pick out an example to be send to test samples folder
    test_data = next(test_generator)
    (masked, mask), ori = test_data

    def plot_callback(model, path):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""
        
        # Get samples & Display them        
        pred_img = model.predict([masked, mask])
        pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Clear current output and display test images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,:])
            axes[1].imshow(pred_img[i,:,:,:] * 1.)
            axes[2].imshow(ori[i,:,:,:])
            axes[0].set_title('Masked Image')
            axes[1].set_title('Predicted Image')
            axes[2].set_title('Original Image')
                    
            plt.savefig(os.path.join(path, '/img_{}_{}.png'.format(i, pred_time)))
            plt.close()

    # Load the model
    if args.vgg_path:
        model = PConvUnet(vgg_weights=args.vgg_path)
    else:
        model = PConvUnet()
    
    # Loading of checkpoint
    if args.checkpoint:
        if args.stage == 'train':
            model.load(args.checkpoint)
        elif args.stage == 'finetune':
            model.load(args.checkpoint, train_bn=False, lr=0.00005)

    # Fit model
    model.fit_generator(
        train_generator, 
        steps_per_epoch=10000,
        validation_data=val_generator,
        validation_steps=1000,
        epochs=100,  
        verbose=0,
        callbacks=[
            TensorBoard(
                log_dir=os.path.join(args.log_path, args.name+'_phase1'),
                write_graph=False
            ),
            ModelCheckpoint(
                os.path.join(args.log_path, args.name+'_phase1', 'weights.{epoch:02d}-{loss:.2f}.h5'),
                monitor='val_loss', 
                save_best_only=True, 
                save_weights_only=True
            ),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: plot_callback(model, args.test_path)
            ),
            TQDMCallback()
        ]
    )
        