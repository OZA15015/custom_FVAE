"""solver.py"""

import os
import visdom
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils import DataGather, mkdirs, grid2gif
from ops import recon_loss, kl_divergence, permute_dims
from model import FactorVAE1, FactorVAE2, Custom_FactorVAE1, Custom_FactorVAE2, Discriminator, Glove_FactorVAE1
from dataset import return_data
import numpy as np
import matplotlib.pyplot as plt
import pylab
from torch.autograd import Variable

#----以下全て, 再現性関連
import random                                                                                                                                 
# cuDNNを使用しない
seed = 32
torch.backends.cudnn.deterministic = True  
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)  
# cuda でのRNGを初期化  
torch.cuda.manual_seed(seed) 


class Solver(object):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.test_count = 0
        self.pbar = tqdm(total=self.max_iter)

        # Data
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)
    
        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        if args.dataset == 'dsprites':
            self.VAE = FactorVAE1(self.z_dim).to(self.device)
            self.nc = 1
        
        elif args.dataset == 'mnist':
            self.VAE = Custom_FactorVAE2(self.z_dim).to(self.device)
            self.nc = 3

        elif args.dataset == 'load_mnist':
            self.VAE = Custom_FactorVAE2(self.z_dim).to(self.device)
            self.nc = 3
       
        elif args.dataset == 'glove/numpy_vector/300d_wiki.npy': 
            self.VAE = Glove_FactorVAE1(self.z_dim).to(self.device)
            self.nc = 3
        else:
            self.VAE = FactorVAE2(self.z_dim).to(self.device)
            self.nc = 3
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.nets = [self.VAE, self.D]

        # Visdom
        self.viz_on = args.viz_on
        self.win_id = dict(D_z='win_D_z', recon='win_recon', kld='win_kld', acc='win_acc')
        self.line_gather = DataGather('iter', 'soft_D_z', 'soft_D_z_pperm', 'recon', 'kld', 'acc')
        self.image_gather = DataGather('true', 'recon')
        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_ra_iter = args.viz_ra_iter
            self.viz_ta_iter = args.viz_ta_iter
            if not self.viz.win_exists(env=self.name+'/lines', win=self.win_id['D_z']):
                self.viz_init()

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)


    def custom_loss(self, x): #lossは交差エントロピーを採用している, MSEの事例もある
        #https://tips-memo.com/vae-pytorch#i-7, http://aidiary.hatenablog.com/entry/20180228/1519828344のlossを参考 
        mean, var = self.VAE._encoder(x)
        #KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var)) #オリジナル, mean意味わからんけど, あんまり値が変わらないか>ら
        #上手くいくんじゃないか
        #KL = 0.5 * torch.sum(torch.exp(var) + mean**2 - 1. - var)
        KL = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp()) 
        # sumを行っているのは各次元ごとに算出しているため
        #print("KL: " + str(KL))
        z = self.VAE._sample_z(mean, var)
        y = self.VAE._decoder(z)
        #delta = 1e-8
        #reconstruction = torch.mean(torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta)))                                    
        #reconstruction = F.binary_cross_entropy(y, x.view(-1, 784), size_average=False)
        reconstruction = F.binary_cross_entropy(y, x, size_average=False)
        #交差エントロピー誤差を利用して, 対数尤度の最大化を行っている, 2つのみ=(1-x), (1-y)で算出可能
        #http://aidiary.hatenablog.com/entry/20180228/1519828344(参考記事)
        #print("reconstruction: " + str(reconstruction))
        #lower_bound = [-KL, reconstruction]
        #両方とも小さくしたい, クロスエントロピーは本来マイナス, KLは小さくしたいからプラスに変換
        #returnで恐らくわかりやすくするために, 目的関数から誤差関数への変換をしている
        #return -sum(lower_bound)
        return KL + reconstruction


    def train(self):
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:#ここで読み込んでいる？
                self.global_iter += 1
                self.pbar.update(1)
                if self.dataset == 'mnist':
                     x_true1 =  x_true1.view(x_true1.shape[0], -1)
                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                x = x_true1.view(x_true1.shape[0], -1) #custom
                vae_recon_loss = self.custom_loss(x) / self.batch_size #custom
                #vae_recon_loss = recon_loss(x_true1, x_recon) #復元誤差, 交差エントロピー誤差
                #vae_kld = kl_divergence(mu, logvar)
                D_z = self.D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean() #恐らく, discriminatorのloss

                #vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss
                vae_loss = vae_recon_loss + self.gamma*vae_tc_loss 
                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()
                x_true2 = x_true2.to(self.device)
                z_prime = self.VAE(x_true2, no_dec=True) #trueにすることで潜在空間に写像した状態のデータを獲得?
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones)) #GANのdiscriminatorっぽい？偽物と本物
                #そのため誤差の部分が0と1になっているはず！zerosとonesの部分

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()

                #if self.global_iter%self.print_iter == 0:
                #    self.pbar.write('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                #        self.global_iter, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))
                if self.test_count % 547 == 0:
                    #self.pbar.write('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                        #self.global_iter, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))
                    self.pbar.write('[{}] vae_recon_loss:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                        self.global_iter, vae_recon_loss.item(), vae_tc_loss.item(), D_tc_loss.item()))  
                    self.test_count = 0
                
                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.viz_on and (self.global_iter%self.viz_ll_iter == 0):
                    soft_D_z = F.softmax(D_z, 1)[:, :1].detach()
                    soft_D_z_pperm = F.softmax(D_z_pperm, 1)[:, :1].detach()
                    D_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_pperm < 0.5).sum()).float()
                    D_acc /= 2*self.batch_size
                    self.line_gather.insert(iter=self.global_iter,
                                            soft_D_z=soft_D_z.mean().item(),
                                            soft_D_z_pperm=soft_D_z_pperm.mean().item(),
                                            recon=vae_recon_loss.item(),
                                            #kld=vae_kld.item(),
                                            acc=D_acc.item())

                if self.viz_on and (self.global_iter%self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ra_iter == 0):
                    self.image_gather.insert(true=x_true1.data.cpu(),
                                             recon=F.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ta_iter == 0):
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        #self.visualize_traverse(limit=3, inter=2/3)
                        print("ignore")

                if self.global_iter >= self.max_iter:
                    out = True
                    break
                self.test_count += 1

        self.pbar.write("[Training Finished]")
        torch.save(self.VAE.state_dict(), "model1/0531_128_2_gamma2.pth")
        self.pbar.close()

    def visualize_recon(self):
        data = self.image_gather.data
        true_image = data['true'][0]
        recon_image = data['recon'][0]

        true_image = make_grid(true_image)
        recon_image = make_grid(recon_image)
        sample = torch.stack([true_image, recon_image], dim=0)
        self.viz.images(sample, env=self.name+'/recon_image',
                        opts=dict(title=str(self.global_iter)))

    def visualize_line(self):
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon = torch.Tensor(data['recon'])
        kld = torch.Tensor(data['kld'])
        D_acc = torch.Tensor(data['acc'])
        soft_D_z = torch.Tensor(data['soft_D_z'])
        soft_D_z_pperm = torch.Tensor(data['soft_D_z_pperm'])
        soft_D_zs = torch.stack([soft_D_z, soft_D_z_pperm], -1)

        self.viz.line(X=iters,
                      Y=soft_D_zs,
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=iters,
                      Y=recon,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=iters,Y=D_acc,
                      env=self.name+'/lines',
                      win=self.win_id['acc'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        '''
        self.viz.line(X=iters,
                      Y=kld,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))
        '''

    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit+0.1, inter)
        random_img = self.data_loader.dataset.__getitem__(0)[1]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        if self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                 'random':random_img_z}

        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'random':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            self.viz.images(samples, env=self.name+'/traverse',
                            opts=dict(title=title), nrow=len(interpolation))

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            mkdirs(output_dir)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(str(os.path.join(output_dir, key+'*.jpg')),
                         str(os.path.join(output_dir, key+'.gif')), delay=10)

        self.net_mode(train=True)

    def viz_init(self):
        zero_init = torch.zeros([1])
        self.viz.line(X=zero_init,
                      Y=torch.stack([zero_init, zero_init], -1),
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['acc'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))
    
    def senzai_view(self, z, label):
        plt.figure(figsize=(10, 10))
        plt.scatter(z[:, 0], z[:, 1], marker='.', c=label, cmap=pylab.cm.jet)
        plt.colorbar()
        plt.grid()
        plt.title('oza_FVAE_2dimention')
        plt.savefig('FVAE0531_128_2_gamma2_senzai.png')

    def load_model(self):
        self.VAE.load_state_dict(torch.load("model1/0531_128_20_gamma2.pth", map_location=self.device))
        for data, label in self.data_loader:
            data = data.to(self.device)
            data = data.view(data.shape[0], -1)
            label = label.detach().numpy()
            break
        
        n = 10
        x_recon, mu, logvar, z = self.VAE(data)
        z = Variable(z, volatile=True).cpu().numpy()
        data = Variable(data, volatile=True).cpu().numpy()
        x_recon = Variable(x_recon, volatile=True).cpu().numpy()
        '''
        plt.figure(figsize=(10, 10))
        plt.scatter(z[:, 0], z[:, 1], marker='.', c=label, cmap=pylab.cm.jet)
        plt.colorbar()
        plt.grid()
        plt.savefig('FVAE0528_128_2_senzai.png')
        '''
        if self.z_dim == 2:
            self.senzai_view(z, label)
        plt.figure(figsize=(12, 6))
        for i in range(n):
            ax = plt.subplot(3, n, i+1)
            if i == 1:
                plt.title('Original MNIST')
            plt.imshow(data[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(3, n, i+1+n)
            if i == 1:
                plt.title('FVAE_Reconstruction MNIST(20dim)')
            plt.imshow(x_recon[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("FVAE0531_128_20_gamma2_recon.png")
        plt.show()
        plt.close()
