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


EXTENSIONS_IMG = ['.jpg']
EXTENSIONS_PT = ['.ply']


def is_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMG)


def is_pt(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_PT)


def read_ply_xyzrgbnormal(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert (os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['cx']
        vertices[:, 4] = plydata['vertex'].data['cy']
        vertices[:, 5] = plydata['vertex'].data['depth']

    return vertices

def read_ply_xyzrgbnormal2(filename):
    assert (os.path.isfile(filename))
    ply_xyz = o3d.io.read_point_cloud(filename) # Read the point cloud
    ply_cxcydepth = meshio.read(filename) # read cx cy depth

    ply_xyz=np.asarray(ply_xyz.points,dtype=np.float32)



    vertices = np.zeros(shape=[ply_xyz.shape[0], 6], dtype=np.float32)
    vertices[:, 0] = ply_xyz[:,0]
    vertices[:, 1] = ply_xyz[:,1]
    vertices[:, 2] = ply_xyz[:,2]
    vertices[:, 3] = ply_cxcydepth.point_data['cx']
    vertices[:, 4] = ply_cxcydepth.point_data['cy']
    vertices[:, 5] = ply_cxcydepth.point_data['depth']

    return vertices

def read_ply_xyzrgbnormal3(filename):
    assert (os.path.isfile(filename))
    ply_xyz = o3d.io.read_point_cloud(filename) # Read the point cloud
    ply_xyz=np.asarray(ply_xyz.points,dtype=np.float32)

    a=[]
    k=0
    with open(filename) as f :
        for j in f:
            k+=1
            if k > 16 :
                #print('j : ',j.split(' '))
                a.append([(float(j.split(' ')[-5])),(float(j.split(' ')[-4])),(float(j.split(' ')[-3]))])
    a=np.asarray(a,dtype=np.float32)


    #print(filename)
    #print(a.shape)
    #print(a)
    #print(ply_xyz.shape)

    vertices = np.zeros(shape=[ply_xyz.shape[0], 6], dtype=np.float32)

    vertices[:, 0] = ply_xyz[:,0]
    vertices[:, 1] = ply_xyz[:,1]
    vertices[:, 2] = ply_xyz[:,2]
    vertices[:, 3] = a[:,0]
    vertices[:, 4] = a[:,1]
    vertices[:, 5] = a[:,2]

    return vertices


class Dataset(data.Dataset):

    def __init__(self, root, phase='train', feature_type='full'):
        self.feature_type = feature_type
        self.phase = phase
        self.img_files = []
        self.point_files = []

        # assert self.feature_type in ['rgb', 'full'], "The feature type must be rgb or full."
        assert self.feature_type in ['rgb', 'full','pt'], "The feature type must be rgb or full."

        img_files = [os.path.join(dp, f + f.split('_')[2]) for dp, dn, fn in os.walk(
            os.path.expanduser(root)) for f in fn if is_img(f)]
        point_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(root)) for f in fn if is_pt(f)]

        assert (len(img_files) == len(point_files))

        self.img_files.extend(img_files)
        self.point_files.extend(point_files)

        self.img_files.sort()
        self.point_files.sort()

        # for RGB_scale
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                # T.Resize((192, 256), interpolation=2),
                T.RandomCrop((128,128)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                # T.Resize((192, 256), interpolation=2),
                T.CenterCrop((128,128)),
                T.ToTensor(),
                normalize
            ])

        print(self.feature_type)

    def __getitem__(self, index):

        img = self.img_files[index]
        ply = self.point_files[index]

        img_path = img[:-3]
        data = Image.open(img_path)
        data = self.transforms(data)
        # data = data.permute(0, 2, 1)
        label = np.int32(img[-3:])

        # PLY 확장자 point cloud, depth
        # real = read_ply_xyzrgbnormal(ply)

        # real = read_ply_xyzrgbnormal2(ply)
        # real = read_ply_xyzrgbnormal3(ply)

        # proj_depth = np.full((256, 192), -1,
        #                      dtype=np.float32)
        # proj_xyz = np.full((3, 256, 192), -1,
        #                    dtype=np.float32)
        #
        # for i in range(len(real)):
        #     proj_depth[int(real[i, 3]), int(real[i, 4])] = real[i, 5]
        #     proj_xyz[0, int(real[i, 3]), int(real[i, 4])] = real[i, 0]
        #     proj_xyz[1, int(real[i, 3]), int(real[i, 4])] = real[i, 1]
        #     proj_xyz[2, int(real[i, 3]), int(real[i, 4])] = real[i, 2]
        #
        # data[:,np.where(proj_depth>0.5)[0],np.where(proj_depth>0.5)[1]] = 0
        # proj_xyz[:,np.where(proj_depth>0.5)[0], np.where(proj_depth>0.5)[1]] = 0
        # proj_depth[np.where(proj_depth>0.5)[0], np.where(proj_depth>0.5)[1]] = 0
        #
        # proj_depth = torch.from_numpy(proj_depth)
        # proj_depth = proj_depth.unsqueeze(0)
        # proj_xyz = torch.from_numpy(proj_xyz)
        # proj = torch.cat((data, proj_depth), dim=0)
        # proj = torch.cat((proj, proj_xyz), dim=0)

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
        return len(self.img_files)

