import re

import torch
from torch import nn

from aimnet.modules import ConfiguredModule


class AIMNetMT(ConfiguredModule):

    def forward(self, coord, number_idx):
        gr, ga = self.aev(coord)
        aims = self.aimnet(gr, ga, number_idx)
        d = dict((out_name, getattr(self, out_name)(aims).squeeze(-1))
                    for out_name in self.output_modules)
        return d

    def scale_outputs(self, outputs):
        for name, val in outputs.items():
            scale = getattr(self, 'scale_' + name, None)
            if scale is not None:
                outputs[name] = val.to(scale.dtype) * scale
        return outputs

    def shift_outputs(self, outputs, numbers):
        for name, val in outputs.items():
            shifts = getattr(self, 'shift_' + name, None)
            if shifts is not None:
                outputs[name] = val.to(shifts.dtype) + shifts[..., numbers]
        return outputs


class EnsembledAIMNetMT(AIMNetMT):

    def __init__(self, **config):
        super().__init__(**config)

        # list on ansamble members
        aimnets = list()
        for name, module in list(self.named_children()):
            if not re.match('aimnet_\d+$', name):
                continue
            aimnets.append(self._modules.pop(name))
        self.aimnets = torch.nn.ModuleList(aimnets)

        # take output_modules names from first model (should be the same for all ensemble members)
        self.output_modules = self.aimnets[0].output_modules

        # take aev from first model (should be the same for all ensemble members)
        for aimnet in self.aimnets[::-1]:
            aev = aimnet._modules.pop('aev')
        self.add_module('aev', aev)

        self._update_ensemble()

    def _update_ensemble(self):
        # take scales from first model (should be the same for all ensemble members)
        for name in self.output_modules:
            name = 'scale_' + name
            scale = None
            for aimnet in list(self.aimnets)[::-1]:
                if hasattr(aimnet, name):
                    scale = aimnet._parameters[name]
            if scale is not None:
                self.register_parameter(name, scale)

        # average shifts
        for name in self.output_modules:
            name = 'shift_' + name
            if hasattr(self.aimnets[0], name):
                avgshift = torch.stack([a._parameters[name] for a in self.aimnets], dim=0).mean(0)
                self.register_parameter(name, nn.Parameter(avgshift, requires_grad=avgshift.requires_grad))

    def forward(self, coord, number_idx):
        gr, ga = self.aev(coord)
        outputs = list()
        for mod in self.aimnets:
            aims = mod.aimnet(gr, ga, number_idx)
            outputs.append(dict((out_name, getattr(mod, out_name)(aims).squeeze(-1))
                                for out_name in self.output_modules))
        avg_outputs = dict()
        for k in outputs[0].keys():
            avg_outputs[k] = torch.stack([o[k] for o in outputs], dim=0).mean(0)

        return avg_outputs


