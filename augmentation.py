import cv2
import numpy as np
import random

def random_augmentations(image):
    aug_images = []

    if random.random() > 0.5:
        aug_images.append(cv2.flip(image, 1))

    if random.random() > 0.5:
        aug_images.append(cv2.GaussianBlur(image, (7, 7), 0))

    if random.random() > 0.5:
        alpha = random.uniform(0.5, 1.5)
        beta = random.randint(-40, 40)
        aug_images.append(cv2.convertScaleAbs(image, alpha=alpha, beta=beta))

    if random.random() > 0.5:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), random.randint(-20, 20), 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        aug_images.append(rotated)

    if len(aug_images) == 0:
        aug_images.append(image)

    return aug_images
