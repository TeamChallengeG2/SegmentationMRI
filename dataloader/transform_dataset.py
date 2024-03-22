from torch.utils.data import Dataset, random_split
from utils.transforms import RandomRotate3D
from utils import load_config

class TransformDataset(Dataset):
    """
    Dataset class which adds augmented data to a base dataset.
    """    
    def __init__(self, base_dataset, config=None):
        """Init method which adds transformation to relevant instance variables.

        Arguments:
            base_dataset: ScoliosisDataset
            config (collections.OrderedDict): OrderedDict config from .json file
        """        
        super(TransformDataset, self).__init__()
        if not config:
            config = load_config("config.json")  

        self.base = base_dataset
        self.angle = config["dataloader"]["rotation_angle"] # Rotation angle range
        self.transforms = [None] # List of transformations
        self.length = len(base_dataset) # Starting length dataset
        if self.angle: # Add RandomRotate3D to self.transformations if enabled
            self.augment_all(RandomRotate3D((-self.angle, self.angle), axes=(0,1)))

    def __len__(self):
        """Returns dataset length including augmentation."""
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
        index_transforms = index // len(self.base)     # Determine which transform 
        transform = self.transforms[index_transforms]
    
        if index >= len(self.base):                    # Determine index in baseset
            index = (index - len(self.base)) % len(self.base)
            
        img, mask, header, filename = self.base[index] # Retrieve items from baseset

        if transform is not None:                      
            img, mask = transform(img, mask, index)    # Rotate

        return img, mask, header, filename
  
    def augment_all(self, transform):
        """
        Increases dataset length and saves transform.

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

if __name__=="__main__":
    from dataloader import scoliosis_dataset
    import matplotlib.pyplot as plt

    train_set_raw, val_set, test_set = scoliosis_dataset() # Base datasets
    train_set = TransformDataset(train_set_raw) # Augmentation in train dataset only!

    plt.imshow(train_set[-1][0][:,:,10], "gray")