from torch.utils.data import Dataset, random_split
from utils.transforms import RandomRotate3D

class TransformDataset(Dataset):
    """
    Dataset class which adds augmented data to a base dataset.
    """    
    def __init__(self, base_dataset, config):
        """Init method which adds transformation to relevant instance variables.

        Arguments:
            base_dataset: ScoliosisDataset
            config (collections.OrderedDict): OrderedDict config from .json file
        """        
        super(TransformDataset, self).__init__()
        self.base = base_dataset
        self.angle = config["dataloader"]["rotation_angle"]
        self.transforms = [None]
        self.length = len(base_dataset)
        if self.angle:
            self.augment_all(RandomRotate3D((-self.angle, self.angle), axes=(0,1)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """Retrieves item and resamples by index.

        Arguments:
            index (int): index number

        Raises:
            IndexError: index > length of dataset

        Returns:
            img
                Torch image tensor. 
            mask
                Torch mask tensor. 
            header
                collections.OrderedDict
            filename
                filename string
        """
        if index >= self.length:
            f"index should be smaller than {self.length}"
            raise IndexError(f"index should be smaller than {self.length}")
        index_transforms = index // len(self.base)     # determine which transform 
        transform = self.transforms[index_transforms]
    
        if index >= len(self.base):                    # determine index in baseset
            index = (index - len(self.base)) % len(self.base)
            
        img, mask, header, filename = self.base[index] # retrieve items from baseset

        if transform is not None:                      
            img, mask = transform(img, mask, index)    # rotate

        return img, mask, header, filename
  
    def augment_all(self, transform):
        """
        Artificially increases dataset length and saves transform.

        Parameters
        ----------
        transform : torchvision.transforms.v2 module
            A data transformation module from Torchvision

        Returns
        -------
        None.

        """
        self.length += len(self.base)
        self.transforms.append(transform)
        print(f"Augmentation done with (-{self.angle}, {self.angle}). Total images: {self.length}")