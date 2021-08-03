import cv2

from tqdm import tqdm

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.svm import LinearSVC
from tensorflow.keras.applications.inception_v3 import preprocess_input


def evaluate_model(model, inception_model, test_generator, test_images=1000):

    metric_maskratio = []
    metric_psnr = []
    metric_ssim = []
    metric_l1 = []

    mae = tf.keras.losses.MeanAbsoluteError()

    # Real & Fake images, to be evaluated with InceptionV3 at the end
    real_images = []
    fake_images = []

    # Loop through test masks released with paper
    for _ in tqdm(range(test_images)):

        # Pick out image from test generator
        test_data = next(test_generator)
        (masked, mask), true = test_data

        # Save real images
        real_images.append(true[0])
        fake_images.append(model.model((masked, mask)).numpy()[0])

        # Save the mask ratio
        metric_maskratio.append((1-mask).sum() / (3 * 512. * 512))

        # Save metrics
        metric_psnr.append(tf.image.psnr(real_images[-1], fake_images[-1], max_val=1).numpy())
        metric_ssim.append(tf.image.ssim(real_images[-1], fake_images[-1], max_val=1).numpy())
        metric_l1.append(mae(real_images[-1], fake_images[-1]).numpy())

    # Put mask ratios into bins
    metric_maskratio = pd.cut(metric_maskratio, np.linspace(0, 1.0, 11))

    # Put results into a dataframe
    df = pd.DataFrame({
        'maskratio': metric_maskratio,
        'psnr': metric_psnr,
        'ssim': metric_ssim,
        'l1': metric_l1,
    })

    # Calculate means and standard deviations of normal metrics
    means = df.groupby('maskratio').mean()
    stds = df.groupby('maskratio').std()
    print(fake_images[0].shape)

    # Rescale real & fakes
    resize_config = {"dsize": (299, 299), "interpolation": cv2.INTER_CUBIC}
    real_images = np.array([np.array(cv2.resize(img, **resize_config)) for img in real_images])
    fake_images = np.array([np.array(cv2.resize(img, **resize_config)) for img in fake_images])

    # Get inception activations for fake & reals
    real_images = inception_model.predict(preprocess_input(real_images), verbose=1, batch_size=8)
    fake_images = inception_model.predict(preprocess_input(fake_images), verbose=1, batch_size=8)

    # Calculate P-IDS for each mask ratio bin
    unique_ratios = np.unique(metric_maskratio)
    for ratio in tqdm(unique_ratios):

        # Get the index
        idx = metric_maskratio == ratio

        # Fit a Linear SVM
        svm = LinearSVC(dual=False)
        svm_inputs = np.concatenate([real_images[idx], fake_images[idx]])
        svm_targets = np.array([1] * real_images[idx].shape[0] + [0] * fake_images[idx].shape[0])
        svm.fit(svm_inputs, svm_targets)

        # Score based on decision boundary
        real_outputs = svm.decision_function(real_images[idx])
        fake_outputs = svm.decision_function(fake_images[idx])

        # Calculate Paired Inception Descriptive Score
        means.loc[ratio, 'p-ids'] = np.mean(fake_outputs > real_outputs)

    return means, stds
