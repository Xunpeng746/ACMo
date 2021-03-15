import math
import torch
from torch.optim.optimizer import Optimizer


class Acutum_OT(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Acutum_OT, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Acutum_OT, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Autum does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # make gradient and momentum sharp
                grad_norm = torch.norm(grad.reshape(-1, ))
                moment_norm = torch.norm(exp_avg.reshape(-1, ))

                coeff = grad_norm.div(moment_norm.add(group['eps']))
                if coeff.item() > 1:
                    coeff = torch.tensor(1.0)
                exp_avg.mul_(coeff).add_(grad).mul_(0.5)
                step_size = group['lr']
                p.data.add_(exp_avg.mul(-step_size))
                # coeff = grad_norm.div(moment_norm.add(group['eps'])
                # if coeff.item() >1:
                #    coeff = torch.tensor(1.0)

                # exp_avg.mul_(coeff).add_(grad)
                # step_size = group['lr']

                # p.data.add_(exp_avg.mul(-step_size))

        return loss
