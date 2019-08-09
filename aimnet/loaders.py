import yaml
import os
import torch
from .modules import get_module


def _process_includes(obj, basepath):
    if isinstance(obj, list):
        for v in obj:
            _process_includes(v, basepath)
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (list, dict)):
                _process_includes(v, basepath)
            if k == '_include':
                obj.update(load_yaml(v, basepath))
                obj.pop('_include')


def load_yaml(yaml_file, basepath=''):
    yaml_file = os.path.join(basepath, yaml_file)
    with open(yaml_file, 'r') as f:
        obj = yaml.load(f, Loader=yaml.SafeLoader)
    _process_includes(obj, os.path.dirname(yaml_file))
    return obj


def module_from_yaml(yaml_file):
    config = load_yaml(yaml_file)
    module = get_module(config['module'])(**config.get('kwargs', {}))
    return module


def _load_mt(mod, mt_state):
    for o in mod.output_modules:
        getattr(mod, o).load_state_dict(
            dict((k[len(o) + 1:], v) for k, v in mt_state.items() if k.startswith(o + '.')))


def load_AIMNetMT(cv=0):
    dirname = os.path.abspath(os.path.dirname(__file__))
    dirname = os.path.join(dirname, 'pretrained', 'aimnet')
    yaml_file = os.path.join(dirname, 'aimnet_mt.yaml')

    mod = module_from_yaml(yaml_file)

    aimnet_state = torch.load(os.path.join(dirname, f'aimnet_cv{cv}.pt'))
    mod.aimnet.load_state_dict(aimnet_state)

    mt_state = torch.load(os.path.join(dirname, f'aimnet_mt_cv{cv}.pt'))
    _load_mt(mod, mt_state)

    return mod


def load_AIMNetSMD(cv=0):
    dirname = os.path.abspath(os.path.dirname(__file__))
    dirname = os.path.join(dirname, 'pretrained', 'aimnet')
    yaml_file = os.path.join(dirname, 'aimnet_e.yaml')

    mod = module_from_yaml(yaml_file)

    aimnet_state = torch.load(os.path.join(dirname, f'aimnet_cv{cv}.pt'))
    mod.aimnet.load_state_dict(aimnet_state)

    mt_state = torch.load(os.path.join(dirname, f'aimnet_smd_cv{cv}.pt'))
    _load_mt(mod, mt_state)

    return mod


def load_AIMNetMT_ens():
    dirname = os.path.abspath(os.path.dirname(__file__))
    dirname = os.path.join(dirname, 'pretrained', 'aimnet')
    yaml_file = os.path.join(dirname, 'aimnet_mt_ens.yaml')
    ens_mod = module_from_yaml(yaml_file)

    for cv, mod in enumerate(ens_mod.aimnets):
        pt_file = os.path.join(dirname, f'aimnet_cv{cv}.pt')
        mod.aimnet.load_state_dict(torch.load(pt_file))

        mt_state = torch.load(os.path.join(dirname, f'aimnet_mt_cv{cv}.pt'))
        _load_mt(mod, mt_state)

    return ens_mod


def load_AIMNetSMD_ens():
    dirname = os.path.abspath(os.path.dirname(__file__))
    dirname = os.path.join(dirname, 'pretrained', 'aimnet')
    yaml_file = os.path.join(dirname, 'aimnet_e_ens.yaml')
    ens_mod = module_from_yaml(yaml_file)

    for cv, mod in enumerate(ens_mod.aimnets):
        pt_file = os.path.join(dirname, f'aimnet_cv{cv}.pt')
        mod.aimnet.load_state_dict(torch.load(pt_file))

        mt_state = torch.load(os.path.join(dirname, f'aimnet_smd_cv{cv}.pt'))
        _load_mt(mod, mt_state)

    return ens_mod
