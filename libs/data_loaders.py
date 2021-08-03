import copy

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.preprocessing.image import NumpyArrayIterator


class InpaintingDirectoryIterator(DirectoryIterator):
    """We overwrite this class in order to get a thread-safe iterator"""

    def _get_batches_of_transformed_samples(self, index_array):

        # Get augmentend image samples
        ori, target = super()._get_batches_of_transformed_samples(index_array)

        # Get masks for each image sample
        mask = np.stack([self.image_data_generator.mask_generator.sample() for _ in range(ori.shape[0])], axis=0)

        # Apply masks to all image sample
        masked = copy.deepcopy(ori)
        masked[mask==0] = 1

        return [masked, mask], ori


class InpaintingNumpyArrayIterator(NumpyArrayIterator):
    """We overwrite this class in order to get a thread-safe iterator"""

    def _get_batches_of_transformed_samples(self, index_array):

        # Get augmentend image samples
        ori = super()._get_batches_of_transformed_samples(index_array)

        # Get masks for each image sample
        mask = np.stack([self.image_data_generator.mask_generator.sample() for _ in range(ori.shape[0])], axis=0)

        # Apply masks to all image sample
        masked = copy.deepcopy(ori)
        masked[mask==0] = 1

        return [masked, mask], ori


class AugmentingDataGenerator(ImageDataGenerator):

    def __init__(self, mask_generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_generator = mask_generator

    def flow_from_directory(self, directory, **kwargs):
        return InpaintingDirectoryIterator(directory, self, **kwargs)



class DataGenerator(ImageDataGenerator):

    def __init__(self, mask_generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_generator = mask_generator

    def flow(self, x, y=None, **kwargs):
        return InpaintingNumpyArrayIterator(x, y, self, **kwargs)