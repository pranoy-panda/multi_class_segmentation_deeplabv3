from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
from torchvision import transforms, utils

class SegDataset(Dataset):
    """Segmentation Dataset"""
 
    def __init__(self, root_dir, imageFolder, maskFolder, class_id_list, transform=None, seed=None, fraction=None, subset=None, imagecolormode='rgb', maskcolormode='grayscale'):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed: Specify a seed for the train and test split
            fraction: A float value from 0 to 1 which specifies the validation split fraction
            subset: 'Train' or 'Test' to select the appropriate set.
            imagecolormode: 'rgb' or 'grayscale'
            maskcolormode: 'rgb' or 'grayscale'
        """
        self.class_id_list = class_id_list
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert(imagecolormode in ['rgb', 'grayscale'])
        assert(maskcolormode in ['rgb', 'grayscale'])
 
        self.imagecolorflag = self.color_dict[imagecolormode]
        self.maskcolorflag = self.color_dict[maskcolormode]
        self.root_dir = root_dir
        self.transform = transform
        if not fraction: # the dataset is not to be divided into train and test/validation
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))
        else:
            assert(subset in ['Train', 'Test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, imageFolder, '*'))))
            self.mask_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, maskFolder, '*'))))
            if seed:
                np.random.seed(seed)
            indices = np.arange(len(self.image_list))
            np.random.shuffle(indices)
            self.image_list = self.image_list[indices]
            self.mask_list = self.mask_list[indices]
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list)*(1-self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list)*(1-self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list)*(1-self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list)*(1-self.fraction))):]
 
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self, idx):
        '''
        As we are dealing with multi-class Segmentation therefore, we need to extract k binary masks
        from the mask image(k = number of classes).
        So, mask for each image would be a tensor of shape (k,W,H)
        '''
        w = 256
        h = 256
        
        img_name = self.image_names[idx]
        image = cv2.imread(img_name)
        image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)
        
        msk_name = self.mask_names[idx]
        mask = cv2.imread(msk_name, 0) # grayscale
        mask = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA)
        
        # lets generate binary masks
        mask_list = []

        for class_id in self.class_id_list:
            mask_list.append(mask == class_id)  
        
        # convert mask_list into a numpy array
        mask_list = np.array(mask_list)
        #print(mask_list.shape)
        print(image.shape)
        sample = {'image': image, 'mask': mask_list}
 
        if self.transform:
            sample = self.transform(sample)
 
        return sample

class Resize(object):
    """Resize image and/or masks."""
 
    def __init__(self, imageresize, maskresize):
        self.imageresize = imageresize
        self.maskresize = maskresize
 
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        '''
        using the transpose() function:
        say the dimension of a matrix M is (a,b,c) and we want to change the shape to (b,a,c)
        then we use the transpose() func as follows: np.transpose(M,(1,0,2)) the indices mean the old dimensions
        '''
        # below channel dim is pushed to the end so that the we can reshape the width and height dimensions
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0) 
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        # perform resize operation
        mask = cv2.resize(mask, self.maskresize, cv2.INTER_AREA)
        image = cv2.resize(image, self.imageresize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1) # revert back to (width,height,channels) as dim 2 earlier was channels
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)
 
        return {'image': image,
                'mask': mask} # returning a dictionary
 
 
class ToTensor(object):
    """Convert ndarrays in sample to Tensors.(For training the network)"""
 
    def __call__(self, sample, maskresize=None, imageresize=None):
        image, mask = sample['image'], sample['mask']
        if len(mask.shape) == 2:
            mask = mask.reshape((1,)+mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}
 
 
class Normalize(object):
    '''Normalize image: This just divides the image pixels by 255 to make them fall in the range of 0 to 1'''
 
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)/255}


def get_dataloader_sep_folder(data_dir, class_id_list, imageFolder='Image', maskFolder='Mask', batch_size=4):
    """
        Create Train and Test dataloaders from two separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
    """
    data_transforms = {
        'Train': transforms.Compose([ToTensor(), Normalize()]),
        'Test': transforms.Compose([ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegDataset(root_dir=os.path.join(data_dir, x),
                                    transform=data_transforms[x], maskFolder=maskFolder, imageFolder=imageFolder,class_id_list = class_id_list)
                      for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=2)
                   for x in ['Train', 'Test']}
    return dataloaders


def get_dataloader_single_folder(data_dir, class_id_list, imageFolder='Images', maskFolder='Masks', fraction=0.2, batch_size=4):
    """
        Create training and testing dataloaders from a single folder.
    """
    data_transforms = {
        'Train': transforms.Compose([ToTensor(), Normalize()]),
        'Test': transforms.Compose([ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=100, fraction=fraction, subset=x, transform=data_transforms[x],class_id_list = class_id_list)
                      for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=2)
                   for x in ['Train', 'Test']}
    return dataloaders