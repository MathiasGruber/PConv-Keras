from random import randint, seed
import itertools
import numpy as np
import cv2


def random_mask(height, width, channels=3, rand_seed=None):
    """Generates a random irregular mask with lines, circles and elipses"""    
    
    img = np.zeros((height, width, channels), np.uint8)

    # Seed for reproducibility
    seed(rand_seed)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")
    
    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
    return 1-img


def load_mask(filepath, height, width, rotation=None, dilation=None, cropping=None):
    """Loads a mask from disk, and optionally augments it"""
    
    # Read image
    mask = cv2.imread(filepath)
    
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
        x = np.random.randint(0, mask.shape[1] - width)
        y = np.random.randint(0, mask.shape[0] - height)
        mask = mask[y:y+height, x:x+width]

    return (mask > 1).astype(np.uint8)
