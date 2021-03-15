import math
import torch
from torch.optim.optimizer import Optimizer


class Acutum_Lipschitz(Optimizer):

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
        super(Acutum_Lipschitz, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Acutum_Lipschitz, self).__setstate__(state)

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
            debug_idx = 1
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
                    state['mmt_t'] = torch.zeros_like(p.data)
                    state['llt_t'] = torch.zeros_like(p.data)
                    state['var_pre'] = torch.zeros_like(p.data)

                mmt_t, llt_t, var_pre = state['mmt_t'], state['llt_t'], state['var_pre']

                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # calculate mmt_now
                var_delta = (p.data).sub(var_pre)
                if (debug_idx==2):
                    print (var_delta)
                    print ("++++++++++++++++++++++++++++++++++++++++++")
                mmt_prime = torch.addcmul(mmt_t, 1.0, llt_t, var_delta)
                if (debug_idx == 2):
                    print (mmt_t)
                    print ('---------------')
                    print (llt_t)
                    print ('---------------')
                    print (mmt_prime)
                    print ('******************************************')
                grad_norm = torch.norm(grad.reshape(-1, ))
                moment_norm = torch.norm(mmt_prime.reshape(-1, ))
                descent_direct = mmt_prime.mul(grad_norm.div(moment_norm.add(group['eps']))).add_(grad).mul_(0.5)
                step_size = group['lr']

                # update states
                mmt_delta = mmt_prime.sub(mmt_t)
                state['llt_t'] = mmt_delta.div(descent_direct.mul(-step_size).add(group['eps']))
                state['mmt_t'] = descent_direct.mul(1.0)
                state['var_pre'] = p.data.mul(1.0)
                p.data.add_(descent_direct.mul(-step_size))
                debug_idx+=1



        return loss
