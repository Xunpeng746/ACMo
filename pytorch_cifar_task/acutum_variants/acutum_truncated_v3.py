import math
import torch
from torch.optim.optimizer import Optimizer


class Acutum_Truncated_V3(Optimizer):

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
        super(Acutum_Truncated_V3, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Acutum_Truncated_V3, self).__setstate__(state)

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
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                grad_norm = torch.norm(grad.reshape(-1,)) ** 2
                moment_norm = torch.norm(exp_avg.reshape(-1,)) ** 2
                gm_dot = torch.dot(exp_avg.reshape(-1,), grad.reshape(-1,))
                #print (grad_norm.item(), gm_dot.item(), (grad_norm.div(grad_norm.add(gm_dot).add(group['eps']))).item())
                if gm_dot.item() >= 0:
                    des_direct = (exp_avg.add(grad)).mul(grad_norm.div(grad_norm.add(gm_dot).add(group['eps'])))
                else:
                    des_direct = (exp_avg.mul(gm_dot.mul(-1).div(moment_norm.add(group['eps'])))).add_(grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(1 - beta1, grad).div_(bias_correction1)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr'] * math.sqrt(bias_correction2)

                p.data.add_(des_direct.mul(-step_size))

        return loss
