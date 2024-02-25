import torch
import numpy as np
import scipy.ndimage
import random

class RandomRotate3D:
    def __init__(self, angle_range, axes=(0, 1)):
        """
        Initialize the random rotation transform.

        Parameters:
        - angle_range (tuple): A tuple of two floats or integers, specifying the minimum and maximum angles in degrees.
        - axes (tuple): A tuple of two integers, specifying the axes around which to rotate the image.
        """
        self.angle_range = angle_range
        self.axes = axes

    def __call__(self, image, mask, seed_):
        """
        Apply the transform to the given 3D image.

        Parameters:
        - image: A 3D torch tensor of shape (D, H, W).

        Returns:
        - The rotated 3D image as a torch tensor.
        """
        random.seed(seed_)
        angle = random.randint(self.angle_range[0], self.angle_range[1])

        # Ensure image is a NumPy array for rotation.
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
            mask_np = mask.numpy()
        else:
            image_np = image
            mask_np = mask

        rotated_image_np = scipy.ndimage.rotate(image_np, angle, axes=self.axes,
                                                reshape=False)
        rotated_mask_np = scipy.ndimage.rotate(mask_np, angle, axes=self.axes,
                                                order=0, mode="nearest", reshape=False)

        # Convert back to torch tensor if the original input was a tensor.
        if isinstance(image, torch.Tensor):
            rotated_image = torch.from_numpy(rotated_image_np)
            rotated_mask = torch.from_numpy(rotated_mask_np)
        else:
            rotated_image = rotated_image_np
            rotated_mask = rotated_mask_np

        return rotated_image, rotated_mask

