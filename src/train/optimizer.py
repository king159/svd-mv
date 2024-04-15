from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer
from transformers.optimization import Adafactor


def exists(val):
    return val is not None


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}

        super().__init__(params, defaults)

        self.update_fn = update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss


def get_optimizer(
    name: str,
    params,
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    weight_decay: float = 0.0,
) -> Optimizer:
    match name:
        case "adamw":
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        case "lion":
            return Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
        case "adafactor":
            return Adafactor(
                params,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=lr,
                weight_decay=weight_decay,
            )
        case _:
            raise ValueError(f"Optimizer {name} not supported")
