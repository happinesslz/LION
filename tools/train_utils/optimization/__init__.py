from functools import partial

import torch.nn as nn
import torch.nn.parameter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        def named_children(m, prefix):
            if num_children(m[1]):
                mm_list = list()
                for mm in m[1].named_children():
                    mm_list.extend(named_children(mm, prefix + f'.{m[0]}' if prefix != '' else f'{m[0]}'))
                return mm_list
            else:
                mm_list = list()
                for n, _ in m[1].named_parameters():
                    mm_list.append(prefix + f'.{m[0]}.{n}' if prefix != '' else f'{m[0]}.{n}')
                return mm_list

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))

        modules = named_children(('', model), '')

        params = model.named_parameters()
        other_params = list()

        for p in params:
            if p[0] not in modules:
                name = p[0].split('.')
                m = model
                for n in name:
                    if n.isnumeric():
                        m = m[int(n)]
                    else:
                        m = getattr(m, n)
                p = p[1]
                if isinstance(m, torch.nn.parameter.Parameter) and hasattr(m, '_no_weight_decay'):
                    setattr(p, '_no_weight_decay', m._no_weight_decay)
                other_params.append(p)

        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), params=iter(other_params), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
