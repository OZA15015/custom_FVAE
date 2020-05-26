"""ops.py"""

import torch
import torch.nn.functional as F


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
#原著論文のGANアルゴリズム1の部分
    assert z.dim() == 2 #2D arrayならokてこと
    B, _ = z.size() #batch_size default:64
    # zのサイズはbatch_size, z_dimとなる
    #defaultでは(64, 10)となる
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device) #64次元のインデックス, それを乱数でシャッフル
        perm_z_j = z_j[perm] #特定の次元に対して, 64つのデータを上記の乱数で入れ替え
        perm_z.append(perm_z_j)
    return torch.cat(perm_z, 1)
