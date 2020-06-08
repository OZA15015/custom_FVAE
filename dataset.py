"""dataset.py"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.datasets import fetch_openml #MNIST

#----以下全て, 再現性関連
import numpy as np
import random
# cuDNNを使用しない
seed = 32
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# cuda でのRNGを初期化
torch.cuda.manual_seed(seed)

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0

i = 0
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)
        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))
 
    def __getitem__(self, index1):
        index2 = random.choice(self.indices)
 
        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2] #index1と違くしたいため乱数っぽい
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
 
        return img1, img2
 
    def __len__(self):
        return self.data_tensor.size(0)



class MNISTDataset(Dataset):
    def __init__(self, data_tensor=None, transform=None):
        self.mnist = fetch_openml('mnist_784', version=1,)
        self.data_tensor = self.mnist.data.reshape(-1, 28, 28).astype('uint8')
        self.transform = transform
        self.target = self.mnist.target.astype(int)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2] #index1と違くしたいため乱数っぽい
        target1 = torch.from_numpy(np.array(self.target[index1]))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        #return self.data_tensor.size(0)
        return len(self.data_tensor)

class load_MNISTDataset(Dataset):
    
    def __init__(self, data_tensor=None, transform=None, root=None):
        self.mnist = fetch_openml('mnist_784', version=1,)
        self.data_tensor = self.mnist.data.reshape(-1, 28, 28).astype('uint8')
        self.target = self.mnist.target.astype(int)
        self.indices = range(len(self))
        self.transform = transform
            
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        idx2 = random.choice(self.indices)
        data1 = self.data_tensor[idx]
        #data2 = self.data[idx2] 
        target1 = torch.from_numpy(np.array(self.target[idx])) #torch.from_numpy()でTensorに変換
        if self.transform:
            data1 = self.transform(data1)
            
        #sample = data1
        return data1, target1

class GloveDataset(Dataset):
    def __init__(self, root, data_tensor=None, transform=None):
        self.data_tensor = np.load(root)
        self.indices = range(len(self))
        #print(self.data_tensor.shape)
        #print(self.indices)
 
    def __getitem__(self, index1):
        index2 = random.choice(self.indices)
 
        data1 = self.data_tensor[index1]
        data2 = self.data_tensor[index2] 
 
        return data1, data2
 
    def __len__(self): 
        #print(len(self.data_tensor))
        return len(self.data_tensor)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),])
    
    transform2 = transforms.ToTensor() #MNIST

    if name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
    elif name.lower() == 'mnist': #試験的MNIST
        print("Test_MNIST")
        train_kwargs = {'data_tensor':None, 'transform':transform2}
        dset = MNISTDataset
    elif name.lower() == 'load_mnist': #試験的MNIST
        print("load_MNIST")
        train_kwargs = {'data_tensor':None, 'transform':transform2}
        dset = load_MNISTDataset

    elif name.lower() == 'glove/numpy_vector/300d_wiki.npy': #試験的Glove
        print("Train_Glove")
        train_kwargs = {'data_tensor':None, 'transform':None, 'root':'/home/oza/pre-experiment/' + name.lower()}
        dset = GloveDataset

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader
