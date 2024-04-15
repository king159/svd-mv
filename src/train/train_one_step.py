import time
from argparse import Namespace
from typing import Optional

import accelerate
import torch
import torch.nn as nn

from .train_utils import append_dims, rand_log_normal


def train_one_step(
    args: Namespace,
    model: nn.Module,
    step: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    accelerator: accelerate.accelerator.Accelerator,
    batch: dict[str, torch.Tensor],
    unchanged_added_time_ids: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """
    Train the model for one step
    Args:
        args (Namespace): arguments
        model (nn.Module): model
        step (int): current step
        optimizer: optimizer
        lr_scheduler: learning rate scheduler
        accelerator: accelerator
        batch (dict[str, torch.Tensor]): the data batch
        unchanged_added_time_ids (torch.Tensor): unchanged added time ids during the whole training including fps, motion bucket id, noise aug strength
    """
    start_time = time.time()
    logging_dict = {}
    for k, v in batch.items():
        batch[k] = v.to(accelerator.device, args.weight_dtype)
    batch_size, num_frames = batch["vae"].shape[:2]
    with torch.no_grad():
        image_embeddings = model.image_encoder(batch["clip"]).image_embeds.unsqueeze(1)
        clip_time = time.time()
        latents = []
        for vae_inputs in torch.split(
            batch["vae"].flatten(0, 1),
            split_size_or_sections=num_frames,
            dim=0,
        ):
            image_latent = model.vae.encode(vae_inputs).latent_dist.mode()
            latents.append(image_latent)
        latents = torch.cat(latents, dim=0)
        latents = latents.unflatten(0, (batch_size, num_frames)) * model.vae.config.scaling_factor
        conditional_image = batch["vae"][:, 0:1, ...]
        noise_aug_strength = torch.exp(
            torch.randn(
                (batch["vae"].shape[0]),
                device=accelerator.device,
                dtype=args.weight_dtype,
            )
            * args.condition_p_std
            + args.condition_p_mean
        )
        conditional_image += torch.randn_like(conditional_image) * append_dims(
            noise_aug_strength, conditional_image.ndim
        )
        conditional_latents = model.vae.encode(conditional_image.squeeze(1)).latent_dist.mode().unsqueeze(1)

        vae_time = time.time()

    added_time_ids = torch.cat(
        [
            noise_aug_strength.unsqueeze(-1).repeat_interleave(num_frames, dim=1).unsqueeze(-1),
            batch["elevation"].unsqueeze(-1),
        ],
        dim=-1,
    )
    sigmas = rand_log_normal((latents.shape[0],), args.p_mean, args.p_std, accelerator.device, latents.dtype)
    sigmas = append_dims(sigmas, latents.ndim)
    noise = torch.randn_like(
        latents,
        requires_grad=False,
    )
    noisy_latents = latents + noise * sigmas

    c_noise = 0.25 * torch.log(sigmas)
    c_skip = 1 / (sigmas**2 + 1)
    c_in = 1 / (sigmas**2 + 1) ** 0.5
    c_out = -sigmas / (sigmas**2 + 1) ** 0.5

    input_noisy_latents = noisy_latents * c_in
    conditional_latents = torch.repeat_interleave(conditional_latents, input_noisy_latents.shape[1], dim=1)
    if torch.rand(1) < args.cfg_rate:
        conditional_latents = torch.zeros_like(conditional_latents)
        image_embeddings = torch.zeros_like(image_embeddings)

    noise_model_input = torch.cat([input_noisy_latents, conditional_latents], dim=2)

    model_pred = model.unet(
        noise_model_input,
        c_noise.squeeze(),
        encoder_hidden_states=image_embeddings,
        added_time_ids=added_time_ids,
    ).sample

    unet_time = time.time()

    denoised_latents = model_pred * c_out + c_skip * noisy_latents
    loss_weighting = sigmas**2 + 1 / sigmas**2

    loss = torch.mean((denoised_latents.float() - latents.float()) ** 2 * loss_weighting.float())
    logging_dict["loss"] = loss.detach().cpu().item()
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.unet.parameters(), args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    backward_time = time.time()
    logging_dict["time/clip_time"] = clip_time - start_time
    logging_dict["time/vae_time"] = vae_time - clip_time
    logging_dict["time/unet_time"] = unet_time - vae_time
    logging_dict["time/backward_time"] = backward_time - unet_time
    logging_dict["time/one_batch_duration"] = time.time() - start_time
    if accelerator.sync_gradients and args.use_ema:
        model.ema_unet.step(model.unet.parameters())
    if accelerator.is_main_process and args.report_to is not None:
        samples_per_second_per_gpu = (
            args.gradient_accumulation_steps * args.batch_size / logging_dict["time/one_batch_duration"]
        )
        samples_per_second = samples_per_second_per_gpu * args.world_size
        logging_dict.update(
            {
                "samples/s": samples_per_second,
                "samples/s/gpu": samples_per_second_per_gpu,
                "lr": lr_scheduler.get_last_lr()[0],
                "loss": logging_dict["loss"],
            }
        )
        accelerator.log(
            logging_dict,
            step=step,
        )
