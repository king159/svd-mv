from typing import Optional

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def get_lr_scheduler(
    name,
    optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    else:
        raise ValueError("Invalid lr scheduler")
