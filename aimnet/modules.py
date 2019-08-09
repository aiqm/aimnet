from importlib import import_module
import os

import math
import torch
import torch.nn.functional as F
from torch import nn


def get_module(name):
    parts = name.split('.')
    mod, func = '.'.join(parts[:-1]), parts[-1]
    mod = import_module(mod)
    func = getattr(mod, func)
    return func


class ConfiguredModule(nn.Module):

    def __init__(self, **config):
        super().__init__()
        self._configure(**config)

    def _configure(self, **config):
        for item in config.get('parameters', []):
            name = item['name']
            dtype = item.get('dtype')
            if dtype is not None:
                dtype = getattr(torch, dtype)
            parameter = nn.Parameter(torch.tensor(item['value'], dtype=dtype),
                                     requires_grad=item.get('requires_grad', False))
            self.register_parameter(name, parameter)

        for item in config.get('modules', []):
            module = get_module(item['module'])(**item.get('kwargs', {}))
            name = item['name']
            self.add_module(name, module)

        for name, value in config.get('variables', {}).items():
            setattr(self, name, value)

        if not config.get('train', False):
            for p in self.parameters():
                p.requires_grad_(False)

        pt_file = config.get('pt_file', None)
        pt_path = config.get('pt_path', os.getcwd())
        if pt_file is not None:
            self.load_state_dict(torch.load(
                os.path.join(pt_path, pt_file), map_location='cpu'), strict=False)


def elu_init_fn(weight):
    nn.init.normal_(weight, std=math.sqrt(1.55 / weight.shape[1]))


def _construct_mpl_layers(sizes, bias=True, activation='ELU', last_linear=False):
    if activation == 'ELU':
        activation = nn.ELU
        elu_init = True
    else:
        activation = get_module(activation)
        elu_init = False

    layers = list()
    for i in range(1, len(sizes)):
        # linear
        l = nn.Linear(sizes[i - 1], sizes[i], bias)
        l.weight.requires_grad_(True)
        if elu_init:
            elu_init_fn(l.weight)
        if bias:
            nn.init.constant_(l.bias, 0)
            l.bias.requires_grad_(True)
        # test
        nn.init.zeros_(l.weight)
        layers.append(l)
        # activation
        if not (last_linear and i == len(sizes) - 1):
            layers.append(activation())
    return layers


def MLPBlock(sizes, bias=True, activation='ELU', last_linear=False):
    """ Make sequential model"""

    layers = _construct_mpl_layers(
        sizes, bias=bias, activation=activation, last_linear=last_linear)
    return nn.Sequential(*layers)


class ANIAEV(ConfiguredModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attr in ('ShfR', 'ShfA', 'ShfZ'):
            for axis in (0, 0, -1):
                getattr(self, attr).unsqueeze_(axis)

    def radial_aev(self, dmat):
        """ dmat[B, N, N] -> gr[B, N, ShfR, N]
        """
        N = dmat.shape[-1]
        fcr = cosine_cutoff(dmat, self.Rcr)
        fcr[..., torch.eye(N, device=fcr.device, dtype=torch.uint8)] *= 0.0
        shifts = (dmat[:, :, None, :] - self.ShfR).pow(2)
        gr = torch.exp(-self.EtaR * shifts) * fcr[:, :, None, :]
        return gr

    def angular_aev(self, dmat, r_ij):
        """ dmat[B, N, N], r_ij[B, B, N] -> ga[B, N, ShfA * ShfZ, N * (N - 1) // 2]
        """
        B, N = dmat.shape[:2]
        nnmask = triu_mask(N, dmat.device)
        nnsize = N * (N - 1) // 2

        # angular part
        cos_ijk = F.cosine_similarity(
            r_ij[:, :, :, None, :], r_ij[:, :, None, :, :], dim=-1)
        cos_ijk = cos_ijk[:, :, nnmask]
        theta_ijk = torch.acos(cos_ijk.clamp(min=-0.99999, max=0.99999))
        ang_part = 1.0 + torch.cos(theta_ijk[:, :, None, :] - self.ShfZ)
        ang_part = torch.pow(ang_part, self.Zeta)[:, :, :, None, :]

        # radial part
        R_ijk_avg = dmat[:, :, :, None] + dmat[:, :, None, :]
        R_ijk_avg = 0.5 * \
            torch.masked_select(R_ijk_avg, nnmask).view(B, N, nnsize)
        shifts = (R_ijk_avg[:, :, None, :] - self.ShfA).pow(2)
        rad_part = torch.exp(-self.EtaA * shifts)[:, :, None, :, :]

        # cutoff part
        fca = cosine_cutoff(dmat, self.Rca)
        fca[..., torch.eye(N, device=fca.device, dtype=torch.uint8)] *= 0.0
        fca_ijk = fca[:, :, :, None] * fca[:, :, None, :]
        fca_ijk = torch.masked_select(fca_ijk, nnmask).view(B, N, nnsize)

        # combine
        ga = torch.pow(2.0, 1.0 - self.Zeta) * fca_ijk
        ga = ga[:, :, None, None, :] * rad_part * ang_part

        ga = ga.view(B, N, -1, nnsize)

        return ga

    def forward(self, coord):
        r_ij = coord[:, :, None, :] - coord[:, None, :, :]
        dmat = r_ij.norm(p=2, dim=-1)
        gr = self.radial_aev(dmat)
        ga = self.angular_aev(dmat, r_ij)
        return gr, ga


class AIMNetEmbed(ConfiguredModule):

    def __init__(self, **config):
        super().__init__(**config)
        # implemented elements
        self.ntyp = self.implemented_elements.numel()
        anumidx = torch.tensor([-1]).repeat(128)
        for i, n in enumerate(self.implemented_elements):
            anumidx[n] = i
        self.register_parameter(
            'anumidx', torch.nn.Parameter(anumidx, requires_grad=False))
        # embedding vectors
        avf_table = torch.empty(self.ntyp, self.nfeature)
        torch.nn.init.orthogonal_(avf_table)
        # test
        torch.nn.init.zeros_(avf_table)
        # sanity check, nan at index -1
        avf_table = torch.cat([avf_table, torch.tensor(
            float('nan')).repeat(self.nfeature).unsqueeze(0)], dim=0)
        self.register_parameter('afv_table', nn.Parameter(
            avf_table, requires_grad=True))

    def forward(self, gr, ga, numbers_or_afv):

        if numbers_or_afv.dtype == torch.long:
            numbers = numbers_or_afv
            afv = self.afv_table[numbers]
            afv_pair = self._combine_afv_0()
            afv_pair = self.combine_mlp(afv_pair)
            afv_pair = afv_pair[self._idx_triu(numbers)]
        else:
            afv = numbers_or_afv
            afv_pair = self._combine_afv_1(afv)
            afv_pair = self.combine_mlp(afv_pair)

        grv = torch.matmul(gr, afv[:, None, :, :]).flatten(-2, -1)
        gav = torch.matmul(ga, afv_pair[:, None, :, :]).flatten(-2, -1)
        aef = self.embedding_mlp(torch.cat([grv, gav], dim=-1))
        return aef, afv

    def number2idx(self, numbers):
        return self.anumidx[numbers]

    def _combine_afv_0(self):
        """
        Combination of base atomic feature vectors
        [X, T, A] -> [X, T(T+1)//2, 2*A]
        """
        afv = self.afv_table[..., :-1, :].transpose(-2, -1)  # [A, T]
        mask = triu_mask(self.ntyp, afv.device, 0)
        afv1, afv2 = afv.unsqueeze(-1), afv.unsqueeze(-2)
        afv2_s = afv1 + afv2  # [A, T, T]
        afv2_p = afv1 * afv2  # [A, T, T]
        afv2 = torch.cat([afv2_s, afv2_p], dim=-3)  # [2A, T, T]
        afv2 = afv2[..., mask].transpose(-2, -1)  # [T(T+1)//2, 2A]
        return afv2

    def _combine_afv_1(self, afv):
        """
        Combination of atomic feature vectors
        [..., N, A] -> [..., N(N-1)//2, A*A]
        """
        N = afv.shape[-2]
        mask = triu_mask(N, afv.device)
        afv = afv.transpose(-2, -1)  # [..., A, N]
        afv1, afv2 = afv.unsqueeze(-1), afv.unsqueeze(-2)
        afv2_s = afv1 + afv2  # [..., A, N, N]
        afv2_p = afv1 * afv2  # [B, A, N, N]
        afv2 = torch.cat([afv2_s, afv2_p], dim=-3)  # [B, 2A, N, N]
        afv2 = afv2[..., mask]  # [B, A, N(N-1)//2]
        afv2 = afv2.transpose(-2, -1)  # [B, N(N-1)//2, 2A]
        return afv2

    def _idx_triu(self, x):
        """
        Unique combinations
        [..., N] -> [..., N]
        """
        i = x * (x + 1) / 2
        idx = x[..., None] * self.ntyp + x[..., None, :] - i[..., None]
        idx = torch.min(idx, idx.transpose(-1, -2))
        mask = triu_mask(idx.shape[-1], idx.device)
        return idx[..., mask]


class AIMNet(ConfiguredModule):
    def forward(self, gr, ga, number_idx):
        output_all_passes = getattr(self, 'output_all_passes', None)

        aims = list()

        aef, afv = self.embed(gr, ga, number_idx)
        for ipass in range(self.npass):
            if ipass > 0:
                afv = afv + self.update_mlp(torch.cat([aef, afv], dim=-1))
                aef, afv = self.embed(gr, ga, afv)
            if ipass == self.npass - 1 or output_all_passes:
                aim = self.interact_mlp(torch.cat([aef, afv], dim=-1))
                aims.append(aim)
        return torch.stack(aims, dim=0)


def triu_mask(n, device, diagonal=1):
    mask = torch.ones(n, n, device=device, dtype=torch.uint8)
    mask = torch.triu(mask, diagonal=diagonal)
    return mask


def cosine_cutoff(d, rc):
    fc = torch.clamp(d / rc, 0, 1)
    fc = 0.5 * torch.cos(math.pi * fc) + 0.5
    return fc

