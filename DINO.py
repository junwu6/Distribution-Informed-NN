import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINO(torch.nn.Module):
    def __init__(self, device, dim, base_s, base_t, nngp_kernels):
        super(DINO, self).__init__()
        self.device = device
        self.length_scale = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.sigma_s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.noise_s_opt = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.noise_t_opt = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        feat_dim = 8
        self.w_s = nn.Linear(feat_dim*2, 1)
        self.w_t = nn.Linear(feat_dim*2, 1)
        self.feat_map = nn.Linear(dim, feat_dim)

        self.k_s_base = torch.tensor(nngp_kernels[0], requires_grad=False, dtype=torch.float).to(device)
        self.k_t_base = torch.tensor(nngp_kernels[1], requires_grad=False, dtype=torch.float).to(device)
        self.k_st_base = torch.tensor(nngp_kernels[2], requires_grad=False, dtype=torch.float).to(device)
        self.base_s = torch.tensor(base_s, requires_grad=False, dtype=torch.float).to(device)
        self.base_t = torch.tensor(base_t, requires_grad=False, dtype=torch.float).to(device)

    def forward(self, source_x, source_y, target_x, target_y, k_ss, k_tt, k_st):
        noise_s_opt = torch.clamp(self.noise_s_opt, min=1e-5, max=1)
        noise_t_opt = torch.clamp(self.noise_t_opt, min=1e-5, max=1)
        K_src = torch.mul(k_ss, self.kernel(source_x, source_x, index1='s', index2='s')) + noise_s_opt * torch.eye(len(source_x)).to(self.device)
        K_tgt = torch.mul(k_tt, self.kernel(target_x, target_x, index1='t', index2='t')) + noise_t_opt * torch.eye(len(target_x)).to(self.device)
        K_cos = torch.mul(k_st, self.kernel(source_x, target_x, index1='s', index2='t'))

        K_src_inv = torch.inverse(K_src)
        mu = K_cos.T @ K_src_inv @ source_y
        cov = K_tgt - K_cos.T @ K_src_inv @ K_cos

        loss = 0.5 * torch.sum(torch.log(torch.diagonal(torch.linalg.cholesky(cov)))) + \
               0.5 * torch.trace((target_y - mu).T @ torch.inverse(cov) @ (target_y - mu)) + 0.5 * len(source_x) * np.log( 2 * np.pi)
        return loss

    def kernel(self, x1, x2, index1, index2):
        w_x1 = self.Kernel_P(x1, index1)
        w_x2 = self.Kernel_P(x2, index2)
        if index1 == 's' and index2 == 's':
            K_base = self.k_s_base
        elif (index1 == 't' and index2 == 't') or (index1 == 'ts' and index2 == 'ts') or (index1 == 'ts' and index2 == 't') or (index1 == 't' and index2 == 'ts'):
            K_base = self.k_t_base
        else:
            K_base = self.k_st_base

        K_P = torch.mm(torch.mm(w_x1, K_base), w_x2.T)
        return K_P

    def Kernel_P(self, x1_ori, index):
        x1 = self.feat_map(x1_ori)
        w_x1 = []
        if index == 's':
            base_vectors = self.feat_map(self.base_s)
        else:
            base_vectors = self.feat_map(self.base_t)
        for i in range(x1.shape[0]):
            if index == 's':
                weight = self.w_s(torch.cat([x1[i, :].repeat(base_vectors.shape[0], 1), base_vectors], dim=1))
            else:
                weight = self.w_t(torch.cat([x1[i, :].repeat(base_vectors.shape[0], 1), base_vectors], dim=1))
            weight = F.elu(weight)
            weight = weight / torch.mean(weight)
            w_x1.append(weight)
        w_x1 = torch.cat(w_x1, dim=1)
        return w_x1.T

    def RBFkernel(self, A, B):
        M = A.shape[0]
        N = B.shape[0]
        A_dots = (A*A).sum(dim=1).reshape((M, 1)) * torch.ones(size=(1, N)).to(self.device)
        B_dots = (B*B).sum(dim=1) * torch.ones(size=(M, 1)).to(self.device)
        dist_matrix = A_dots + B_dots - 2 * torch.mm(A, B.T)
        return (self.sigma_s **2) * torch.exp(-1 / (2 * (self.length_scale ** 2)) * dist_matrix)

    def inference(self, test_x, source_x, source_y, target_x, target_y, k_ss, k_tt, k_st, k_s_ts, k_t_ts):
        noise_s_opt = torch.clamp(self.noise_s_opt, min=1e-5, max=1)
        noise_t_opt = torch.clamp(self.noise_t_opt, min=1e-5, max=1)

        K_src = torch.mul(k_ss, self.kernel(source_x, source_x, index1='s', index2='s')) + noise_s_opt * torch.eye(len(source_x)).to(self.device)
        K_tgt = torch.mul(k_tt, self.kernel(target_x, target_x, index1='t', index2='t')) + noise_t_opt * torch.eye(len(target_x)).to(self.device)
        K_cos = torch.mul(k_st, self.kernel(source_x, target_x, index1='s', index2='t'))
        K = torch.cat([torch.cat([K_src, K_cos], dim=1), torch.cat([K_cos.T, K_tgt], dim=1)], dim=0)
        K_ts = torch.cat([torch.mul(k_s_ts, self.kernel(source_x, test_x, index1='s', index2='ts')),
                          torch.mul(k_t_ts, self.kernel(target_x, test_x, index1='t', index2='ts'))], dim=0)
        K_inv = torch.inverse(K)

        mu_s = K_ts.T @ K_inv @ torch.cat([source_y, target_y], dim=0)
        # cov_s = K_test - K_ts.T @ K_inv @ K_ts
        return mu_s


