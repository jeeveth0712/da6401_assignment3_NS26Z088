"""
utils.py — Noam Learning Rate Scheduler
DA6401 Assignment 3: "Attention Is All You Need"
"""


class NoamScheduler:
    """
    Implements the Noam learning rate schedule from §5.3:

        lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))

    Linearly increases lr for the first warmup_steps steps, then decays
    proportionally to the inverse square root of the step number.

    Args:
        optimizer     : Wrapped optimizer.
        d_model       : Model dimensionality (controls the peak lr scale).
        warmup_steps  : Number of warmup steps (default 4000).
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def step(self):
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _compute_lr(self, step: int) -> float:
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )

    def get_last_lr(self):
        return [self._compute_lr(max(1, self._step))]

    def state_dict(self):
        return {'_step': self._step}

    def load_state_dict(self, state_dict: dict):
        self._step = state_dict['_step']
