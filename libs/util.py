import os
from random import randint, seed
import numpy as np
import cv2


class MaskGenerator():

    def __init__(self,
        height, width, channels=3, rand_seed=None, filepath=None,
        rotation=True, dilation=True, cropping=True, invert=False,
        random_draws=70, target_ratio=None
    ):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width
            channels {int} -- Mask channels (default: {3})
            rand_seed {int} -- Random seed (default: {None})
            filepath {str} -- Path to mask files (default: {None})
            rotation {bool} -- Random rotation (default: {True})
            dilation {bool} -- Random dilation (default: {True})
            cropping {bool} -- Random cropping (default: {True})
            invert {bool} -- Invert mask (default: {False})
            random_draws {int} -- Number of random draws (default: {70})
            target_ratio {float} -- Target ratio of white pixels to total pixels (default: {None})
        """

        # Set parameters
        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath
        self.invert = invert
        self.random_draws = random_draws
        self.target_ratio = target_ratio

        # Specific for loaded masks
        self.rotation = rotation
        self.dilation = dilation
        self.cropping = cropping

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _randomdraw(self, img, size):
        """Draw a random object on the image"""

        # Pick random action
        action = np.random.randint(0, 3)

        # Draw random line
        if action == 0:
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)

        # Draw random circles
        elif action == 1:
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)

        # Draw random ellipsis
        elif action == 2:
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw objects
        if self.target_ratio is None:
            for _ in range(np.random.randint(1, self.random_draws)):
                self._randomdraw(img, size)
        else:
            ratio = 0
            while ratio < self.target_ratio:
                self._randomdraw(img, size)
                ratio = np.mean(img)

        return img if self.invert else 1-img

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        """Loads a mask from disk, and optionally augments it"""

        # Read image
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]))

        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y+self.height, x:x+self.width]

        mask = (mask > 1).astype(np.uint8)

        return 1-mask if self.invert else mask

    def sample(self, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask(rotation=self.rotation, dilation=self.dilation, cropping=self.cropping)
        else:
            return self._generate_mask()
