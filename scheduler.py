from collections.abc import Iterable
from math import log, cos, pi, floor

from torch.optim.lr_scheduler import _LRScheduler


class CyclicCosineDecayLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 init_interval,
                 min_lr,
                 restart_multiplier=None,
                 restart_interval=None,
                 restart_lr=None,
                 last_epoch=-1):
        """
        Initialize new CyclicCosineDecayLR object
        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_interval: (int) - Initial decay cycle interval.
        :param min_lr: (float or iterable of floats) - Minimal learning rate.
        :param restart_multiplier: (float) - Multiplication coefficient for increasing cycle intervals,
            if this parameter is set, restart_interval must be None.
        :param restart_interval: (int) - Restart interval for fixed cycle intervals,
            if this parameter is set, restart_multiplier must be None.
        :param restart_lr: (float or iterable of floats) - Optional, the learning rate at cycle restarts,
            if not provided, initial learning rate will be used.
        :param last_epoch: (int) - Last epoch.
        """

        if restart_interval is not None and restart_multiplier is not None:
            raise ValueError("You can either set restart_interval or restart_multiplier but not both")

        if isinstance(min_lr, Iterable) and len(min_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_lr), len(optimizer.param_groups)))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if init_interval <= 0:
            raise ValueError("init_interval must be a positive number, got {} instead".format(init_interval))

        group_num = len(optimizer.param_groups)
        self._init_interval = init_interval
        self._min_lr = [min_lr] * group_num if isinstance(min_lr, float) else min_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_multiplier = restart_multiplier
        super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self._init_interval:
            return self._calc(self.last_epoch,
                              self._init_interval,
                              self.base_lrs)

        elif self._restart_interval is not None:
            cycle_epoch = (self.last_epoch - self._init_interval) % self._restart_interval
            lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
            return self._calc(cycle_epoch,
                              self._restart_interval,
                              lrs)

        elif self._restart_multiplier is not None:
            n = self._get_n(self.last_epoch)
            sn_prev = self._partial_sum(n)
            cycle_epoch = self.last_epoch - sn_prev
            interval = self._init_interval * self._restart_multiplier ** n
            lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
            return self._calc(cycle_epoch,
                              interval,
                              lrs)
        else:
            return self._min_lr

    def _calc(self, t, T, lrs):
        return [min_lr + (lr - min_lr) * (1 + cos(pi * t / T)) / 2
                for lr, min_lr in zip(lrs, self._min_lr)]

    def _get_n(self, epoch):
        a = self._init_interval
        r = self._restart_multiplier
        _t = 1 - (1 - r) * epoch / a
        return floor(log(_t, r))

    def _partial_sum(self, n):
        a = self._init_interval
        r = self._restart_multiplier
        return a * (1 - r ** n) / (1 - r)
