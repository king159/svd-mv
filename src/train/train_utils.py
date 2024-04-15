import os
import shutil

import safetensors.torch
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def rand_log_normal(
    shape: tuple,
    loc: float = 0.0,
    scale: float = 1.0,
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def gather_different_size_tensor(x: torch.Tensor, accelerator) -> torch.Tensor:
    local_size = torch.tensor(x.shape[0], device=accelerator.device)
    all_sizes = accelerator.gather(local_size)

    max_length = max(all_sizes).item()

    length_diff = max_length - local_size.item()
    if length_diff:
        padding = torch.zeros([length_diff, *x.shape[1:]], device=accelerator.device, dtype=x.dtype)
        padded_tensor = torch.cat([x, padding], dim=0)
    else:
        padded_tensor = x
    all_padded_tensor = accelerator.gather(padded_tensor)
    result = []
    for idx, size in enumerate(all_sizes):
        result.append(all_padded_tensor[idx * max_length : idx * max_length + size])
    return torch.cat(result, dim=0)


def convert_zero3_to_safetensors(
    checkpoint_dir: str,
) -> None:
    """
    convert a folder of deepspeed zero3 checkpoints to safetensors checkpoint
    """
    tag = checkpoint_dir.split("/")[-1]
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir.removesuffix(tag), tag=tag)
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    safetensors.torch.save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
