import math
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, milestones, min_ratio=0., cycle_decay=1., warmup_iters=1000, warmup_factor=1./1000, last_epoch=-1, dataset='something'):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )
        self.milestones = [warmup_iters]+milestones
        self.min_ratio = min_ratio
        self.cycle_decay = cycle_decay
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.dataset = dataset
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
#         self.last_epoch = last_epoch
        
    def get_lr(self):
#         if (self.last_epoch ==0):
#             warmup_factor = self.warmup_factor
#             lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]    
#             return lrs
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
#             alpha = pow(10,self.last_epoch)            
#             warmup_factor = self.warmup_factor *(alpha)
            lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]    

        else:
            cycle = min(bisect_right(self.milestones, self.last_epoch), len(self.milestones)-1)
            # calculate the fraction in the cycle
            fraction = min((self.last_epoch - self.milestones[cycle-1]) / (self.milestones[cycle]-self.milestones[cycle-1]), 1.)
            lrs = [base_lr*self.min_ratio + (base_lr * self.cycle_decay**(cycle-1) - base_lr*self.min_ratio) * (1 + math.cos(math.pi * fraction)) / 2 for base_lr in self.base_lrs] 
        
        if 'kinetics' in self.dataset:
            lrs[-2] =lrs[-2] * 5
            lrs[-1] = lrs[-1] *10
            
        return lrs