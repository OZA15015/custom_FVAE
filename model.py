"""model.py"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

#----以下全て, 再現性関連
import numpy as np
import random
seed = 32
# cuDNNを使用しない  
torch.backends.cudnn.deterministic = True  
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)  
# cuda でのRNGを初期化  
torch.cuda.manual_seed(seed)  



class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAE1(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=10):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        print("eps")
        print(eps.shape)
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()


class FactorVAE2(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
    def __init__(self, z_dim=10):
        super(FactorVAE2, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace = False),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace = False),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace = False),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace = False),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(inplace = False),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(inplace = False),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(inplace = False),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace = False),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace = False),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace = False),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()


class FactorVAE3(nn.Module):
    """Encoder and Decoder architecture for 3D Faces data."""
    def __init__(self, z_dim=10):
        super(FactorVAE3, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()


class Custom_FactorVAE1(nn.Module):
    """Encoder and Decoder architecture for MNIST_Test"""
    def __init__(self, z_dim=10):
        super(Custom_FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, z_dim)
        )
        self.decode = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Linear(200, 28*28),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self, mode='normal'):              
        if mode == 'kaiming':                          
            initializer = kaiming_init                 
        elif mode == 'normal':                         
            initializer = normal_init                  
                                                       
        for block in self._modules:                    
            for m in self._modules[block]:             
                initializer(m)                         
                                                       
    def reparametrize(self, mu, logvar):               
        std = logvar.mul(0.5).exp_()
        #print(std.shape)                 
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)                   
                                                       
    def forward(self, x, no_dec=False):                
        stats = self.encode(x)
        print("stats")
        print(stats.shape)
        quit()
        mu = stats[:, :self.z_dim]                     
        logvar = stats[:, self.z_dim:]                 
        z = self.reparametrize(mu, logvar)
        if no_dec:                                     
            return z.squeeze()                         
        else:                                          
            x_recon = self.decode(z)                   

class Custom_FactorVAE2(nn.Module):
    def __init__(self, z_dim=10):
        super(Custom_FactorVAE2, self).__init__()
        self.z_dim = z_dim
        self.dense_enc1 = nn.Linear(28*28, 200)
        self.dense_enc2 = nn.Linear(200, 100)
        self.dense_encmean = nn.Linear(100, z_dim)
        self.dense_encvar = nn.Linear(100, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 100)
        self.dense_dec2 = nn.Linear(100, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)
        #self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)
        
    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        var = self.dense_encvar(x)
        #var = F.softplus(self.dense_encvar(x))
        return mean, var
    
    def _sample_z(self, mean, var): #普通にやると誤差逆伝搬ができないのでReparameterization Trickを活用
        device = 'cuda'
        epsilon = torch.randn(mean.shape).to(device)
        #return mean + torch.sqrt(var) * epsilon #平均 + episilonは正規分布に従う乱数, torc.sqrtは分散とみなす？平均のルート
        return mean + epsilon * torch.exp(0.5*var)
        # イメージとしては正規分布の中からランダムにデータを取り出している
        #入力に対して潜在空間上で類似したデータを復元できるように学習, 潜在変数を変化させると類似したデータを生成
        #Autoencoderは決定論的入力と同じものを復元しようとする


    def _decoder(self,z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = F.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x, no_dec=False):
        x = x.view(x.shape[0], -1)
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self._decoder(z)
            return x_recon, mean, var, z.squeeze()

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
