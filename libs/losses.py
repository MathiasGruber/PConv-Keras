import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def l1_error(y_true, y_pred):
    """Calculate the L1 loss used in all loss calculations"""
    if K.ndim(y_true) == 4:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
    elif K.ndim(y_true) == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2])
    else:
        raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")


def gram_matrix(x, norm_by_channels=False):
    """Calculate gram matrix used in style loss"""
    
    # Assertions on input
    assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
    assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
    
    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[1], shape[2], shape[3]
    
    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H*W]))
    gram = K.batch_dot(features, features, axes=2)
    
    # Normalize with channels, height and width
    gram = gram /  K.cast(C * H * W, x.dtype)
    
    return gram
        

def total_variation(y_comp):
    """Total variation loss, used for smoothing the hole region, see. eq. 6"""
    a = l1_error(y_comp[:,1:,:,:], y_comp[:,:-1,:,:])
    b = l1_error(y_comp[:,:,1:,:], y_comp[:,:,:-1,:])  
    return a + b