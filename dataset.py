import glob
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Configuration
from config import CONFIG
RAMBO_MOUNT_POINT = CONFIG['rambo_mount_point']




class MobilityDataset(Dataset):
    def __init__(self, test_set=False):
        self.df_dataset = pd.read_csv("train_data.csv")
        self.key = np.fromfile(RAMBO_MOUNT_POINT + "/edata/WICU_DATASET_2014/demoTracking/key.bin", dtype=np.int8)

        #### Split dataset into train/test
        TRAIN_IDS = [0]#,1,2]
        TEST_IDS = [3]
        if test_set == False:
            self.df_dataset = self.df_dataset[self.df_dataset.user_id.isin(TRAIN_IDS)]
        else:
            self.df_dataset = self.df_dataset[self.df_dataset.user_id.isin(TEST_IDS)]




    def __len__(self):
        return len(self.df_dataset)




    def __getitem__(self, idx):
        """ Return a set of images and a label """
        row = self.df_dataset.iloc[idx]
        y = row['mobility_label']
        img_dir = row['image_directory']

        # Get 10 images sampled evenly across all images recorded over a minute
        images = self.sample_ims(img_dir)
        images = self.image_transforms(images)
        return images, y





    def decrypt_image(self, image_file):
        """ Decrypt an image on Rambo """
        fileAsData = np.fromfile(image_file, np.int8)
        bytesRemaining = fileAsData.size
        while bytesRemaining > 0:
            blockStart = fileAsData.size - bytesRemaining
            blockSize = np.minimum(bytesRemaining, self.key.size)
            fileAsData[blockStart:blockStart+blockSize] -= self.key[:blockSize]
            bytesRemaining -= blockSize
        fileInMemory = BytesIO(fileAsData.tostring())
        dataImage = Image.open(fileInMemory)
        return dataImage




    def sample_ims(self, img_dir, num_ims=10):
        """ Sample images from a directory """
        # Get all images
        all_ims = sorted(glob.glob(RAMBO_MOUNT_POINT + img_dir + "/*.jpg"))

        # Sample the images
        sample_idxs = np.linspace(0, len(all_ims), num_ims, endpoint=False, dtype=int)
        sampled_ims = [all_ims[x] for x in sample_idxs]

        # Decrypt images
        return [self.decrypt_image(im) for im in sampled_ims]




    def image_transforms(self, pil_imgs):
        """ Transformations on a list of images """

        # Get random parameters to apply same transformation to all images in list
        color_jitter = transforms.ColorJitter.get_params(.25,.25,.25,.25)
        rotation_param = transforms.RandomRotation.get_params((-15,15))

        # Apply transformations
        images = []
        for pil_img in pil_imgs:
            i = transforms.functional.resize(pil_img, (224,224))
            # if self.train:
            #     i = color_jitter(i)
            #     i = transforms.functional.rotate(i, rotation_param)
            i = transforms.functional.to_tensor(i)
            i = transforms.functional.normalize(i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            images.append(i)
        return torch.stack(images)
