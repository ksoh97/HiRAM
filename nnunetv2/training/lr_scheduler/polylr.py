from torch.optim.lr_scheduler import _LRScheduler
import math


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        # super().__init__(optimizer, current_step if current_step is not None else -1, False)
        super().__init__(optimizer, last_epoch=current_step if current_step is not None else -1)


    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class CosineAnnealingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, cycles: int = 3, eta_min: float = 0, current_step: int = None):
        self.initial_lr = initial_lr
        self.total_steps = max_steps
        self.cycles = cycles
        self.eta_min = eta_min
        self.ctr = 0
        self.steps_per_cycle = max_steps // cycles
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        cycle_step = current_step % self.steps_per_cycle
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_step / self.steps_per_cycle))
        new_lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr