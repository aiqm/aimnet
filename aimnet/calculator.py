import torch
from ase.calculators.calculator import Calculator, all_changes
from ase import units


class AIMNetCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, **kwargs):
        super().__init__(*kwargs)
        self.model = model
        self.device = list(model.parameters())[0].device
        if hasattr(model, 'aimnets'):
            self.number2idx = model.aimnets[0].aimnet.embed.number2idx
        else:
            self.number2idx = model.aimnet.embed.number2idx

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        coord = torch.from_numpy(self.atoms.positions).to(torch.float).to(self.device).unsqueeze(0)
        number_idx = self.number2idx(torch.from_numpy(self.atoms.numbers).to(self.device)).unsqueeze(0)

        if 'forces' in properties:
            coord.requires_grad_(True)

        output = self.model(coord, number_idx)
        output = self.model.scale_outputs(output)

        if 'forces' in properties:
            e = output['energy'].sum(-1)
            output['forces'] = - torch.autograd.grad(e, coord, torch.ones_like(e))[0]

        output = self.model.shift_outputs(output, number_idx)
        output['energy'] = output['energy'].sum(-1)

        for k in ('energy', 'forces'):
            if k in output:
                output[k] *= units.Hartree

        for k, v in output.items():
            output[k] = v.detach().squeeze(0).cpu().numpy()

        self.results = output
 