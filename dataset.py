import os
import time
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T
import numpy as np
from plyfile import PlyData
# import open3d as o3d
# import meshio
import numpy as np


EXTENSIONS = ['.npy']

def is_data(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)



class Dataset(data.Dataset):

    def __init__(self, root, phase='train', feature_type='full'):
        self.feature_type = feature_type
        self.phase = phase
        self.data_files = []


        # assert self.feature_type in ['rgb', 'full'], "The feature type must be rgb or full."
        assert self.feature_type in ['rgb', 'full','pt'], "The feature type must be rgb or full."

        data_files = [os.path.join(dp, f + f.split('_')[1]) for dp, dn, fn in os.walk(
            os.path.expanduser(root)) for f in fn if is_data(f)]

        self.data_files.extend(data_files)

        self.data_files.sort()

        # for RGB_scale
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.ToTensor(),
                # T.Resize((192, 256), interpolation=2),
                T.RandomCrop((128,128)),
                T.RandomHorizontalFlip(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                # T.Resize((192, 256), interpolation=2),
                T.CenterCrop((128,128)),
                normalize
            ])

        #pt 및 depth는 not normalize
        if self.phase == 'train':
            self.transforms2 = T.Compose([
                T.ToTensor(),
                # T.Resize((192, 256), interpolation=2),
                T.RandomCrop((128,128)),
                T.RandomHorizontalFlip()

            ])
        else:
            self.transforms2 = T.Compose([
                T.ToTensor(),
                # T.Resize((192, 256), interpolation=2),
                T.CenterCrop((128,128))
            ])

        print(self.feature_type)

    def __getitem__(self, index):

        data = self.data_files[index]

        data_path = data[:-3]
        # print(data_path)
        fusion = np.load(data_path, allow_pickle=True)

        img = fusion[:,:,0:3]
        points = fusion[:,:,3:]

        #0~1 scaling
        img = img/255

        img = self.transforms(img)
        points = self.transforms2(points)

        label = np.int32(data[-3:])

        proj = torch.cat((img, points), dim=0)

        if self.feature_type == 'rgb':
            return data.float(), label
        elif self.feature_type == 'full':
            return proj.float(), label
        elif self.feature_type == 'pt':
            return proj_xyz.float(), label
        else:
            assert self.feature_type not in ['rgb', 'full'], "The feature type must be rgb or full."


    #         return proj.float(), label
    #         return data.float(), label

    def __len__(self):
        return len(self.data_files)

