import os
from argparse import Namespace

import safetensors.torch
import torch

from ..train.train_utils import convert_zero3_to_safetensors


@torch.no_grad()
def save_every_n_step(model, args: Namespace, accelerator, step: int, lr_scheduler, optimizer) -> None:
    model.eval()
    save_checkpoint_path = os.path.join("model_ckpt", args.exp_name)
    os.makedirs(save_checkpoint_path, exist_ok=True)
    if accelerator.state.deepspeed_plugin.zero_stage != 3:
        if accelerator.is_main_process:
            model = accelerator.unwrap_model(accelerator.unwrap_model(model))
            model.unet.save_pretrained(
                save_directory=save_checkpoint_path,
                state_dict={name: param for name, param in model.named_parameters() if param.requires_grad},
                save_function=safetensors.torch.save_file,
            )
    else:
        # all subprocesses need to call this function when using zero3
        model.save_checkpoint(
            save_checkpoint_path,
            exclude_frozen_parameters=True,
            tag=f"step_{step}",
            save_latest=False,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            convert_zero3_to_safetensors(os.path.join(save_checkpoint_path, f"step_{step}"))
            torch.save(
                {
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "hyper_parameters": args,
                },
                os.path.join(save_checkpoint_path, f"step_{step}", "others.pt"),
            )
    model.train()
