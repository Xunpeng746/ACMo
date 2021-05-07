import torch
import torch.optim
import math
from torch.optim.optimizer import Optimizer
from fairseq.optim import LegacyFairseqOptimizer, register_optimizer

@register_optimizer("acmo")
class FairseqACMo(LegacyFairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = Acutum_Original(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--acmo-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for ACMo optimizer')
        parser.add_argument('--acmo-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for ACMo optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.args.lr[0],
            "betas": eval(self.args.acmo_betas),
            "eps": self.args.acmo_eps,
            "weight_decay": self.args.weight_decay,
        }


class Acutum_Original(Optimizer):

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
        super(Acutum_Original, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Acutum_Original, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        grad_host = None
        moment_host = None

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
                    state['exp_avg'] = torch.zeros_like(p.data)
                exp_avg= state['exp_avg']

                if grad_host is None:
                    grad_host = grad.view(-1)
                else:
                    grad_host = torch.cat((grad_host, grad.view(-1)), dim=-1)

                
                if moment_host is None:
                    moment_host = exp_avg.view(-1)
                else:
                    moment_host = torch.cat((moment_host, exp_avg.view(-1)), dim=-1)
        
        grad_norm = torch.norm(grad_host)
        moment_norm = torch.norm(moment_host)


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

                exp_avg= state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # make gradient and momentum sharp
                # grad_norm = torch.norm(grad.reshape(-1, ))
                # moment_norm = torch.norm(exp_avg.reshape(-1, ))

                exp_avg.mul_(grad_norm.div(moment_norm.add(group['eps']))).add_(grad).mul_(0.9)
                step_size = group['lr']

                p.data.add_(exp_avg.mul(-step_size))

        return loss