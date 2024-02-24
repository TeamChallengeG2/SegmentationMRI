import torch
import numpy as np
import scipy.ndimage


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

    def __call__(self, image):
        """
        Apply the transform to the given 3D image.

        Parameters:
        - image: A 3D torch tensor of shape (D, H, W).

        Returns:
        - The rotated 3D image as a torch tensor.
        """
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])

        # Ensure image is a NumPy array for rotation.
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        else:
            image_np = image

        rotated_image_np = scipy.ndimage.rotate(image_np, angle, axes=self.axes, reshape=False)

        # Convert back to torch tensor if the original input was a tensor.
        if isinstance(image, torch.Tensor):
            rotated_image = torch.from_numpy(rotated_image_np)
        else:
            rotated_image = rotated_image_np

        return rotated_image

