import math
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
from diffusers import StableVideoDiffusionPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..train.my_svd_unet import UNetSpatioTemporalConditionModel


try:
    import wandb
except ImportError:
    pass

from ..train.train_utils import append_dims, gather_different_size_tensor
from . import eval_metrics


def convert_video(vid_tensor: torch.Tensor) -> torch.Tensor:
    # b, t, c, h, w
    video = vid_tensor.detach()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8)
    return video


def denoising(
    pipeline,
    batch,
    accelerator,
    args,
    idx,
    num_frames,
    num_inference_steps,
    unchanged_added_time_ids,
    device,
    num_videos_per_prompt,
    noise_aug_strength,
    min_guidance_scale,
    max_guidance_scale,
    timesteps,
    latent,
    batch_size,
) -> torch.Tensor:
    # Encode input image using CLIP to get image embeddings
    image_embeddings = pipeline._encode_image(
        batch["clip"].float(),
        accelerator.device,
        num_videos_per_prompt,
        pipeline.do_classifier_free_guidance,
    ).to(args.weight_dtype)

    # Encode input image using VAE to get image latents
    image = batch["vae"][:, 0, ...]
    noise = torch.randn_like(image)
    image = image + noise_aug_strength * noise
    image_latents = pipeline._encode_vae_image(
        image, device, num_videos_per_prompt, pipeline.do_classifier_free_guidance
    )
    image_latents = image_latents.to(image_embeddings.dtype)
    image_latents = image_latents.unsqueeze(1).repeat_interleave(num_frames, 1)

    if pipeline.unet.config["projection_class_embeddings_input_dim"] == 768:
        # original pipeline
        added_time_ids = unchanged_added_time_ids.repeat(batch_size, num_frames, 1)
    else:
        added_time_ids = torch.cat(
            [
                unchanged_added_time_ids[-1:].repeat(batch_size, num_frames, 1),
                batch["elevation"].unsqueeze(-1),
            ],
            dim=-1,
        )

    if pipeline.do_classifier_free_guidance:
        added_time_ids = added_time_ids.repeat(2, 1, 1)

    # Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    latents = (
        latent.repeat_interleave(batch_size, 0).to(device, dtype=args.weight_dtype)
    ) * pipeline.scheduler.init_noise_sigma
    # 7. Prepare guidance scale
    guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames)
    guidance_scale = guidance_scale.to(device, latents.dtype)
    guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
    guidance_scale = append_dims(guidance_scale, latents.ndim)
    pipeline._guidance_scale = guidance_scale

    # diffusion process starts
    pipeline._num_timesteps = len(timesteps)
    for t in tqdm(
        timesteps,
        desc=f"Generating video {idx+1}",
        leave=False,
        disable=not accelerator.is_main_process,
    ):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Concatenate image_latents over channels dimension
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

        # predict the noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]

        # perform guidance
        if pipeline.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
    return latents


@torch.no_grad()
def eval_every_n_step(
    model: nn.Module,
    step: int,
    accelerator,
    args: Namespace,
    eval_dataloader: DataLoader,
    latent: torch.Tensor,
) -> None:
    model.eval()
    init_latent = latent.detach().clone().to(device=accelerator.device, dtype=args.weight_dtype)
    # qualitative evaluation
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=accelerator.unwrap_model(model.unet),
        torch_dtype=args.weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    original_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=args.weight_dtype,
        local_files_only=True,
    )
    original_pipeline.unet = UNetSpatioTemporalConditionModel.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=args.weight_dtype,
        subfolder="unet",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )
    original_pipeline = original_pipeline.to(accelerator.device,)
    original_pipeline.set_progress_bar_config(disable=True)

    num_videos_per_prompt = 1
    max_guidance_scale = 3.0
    fps = 6
    noise_aug_strength = torch.tensor([0.02], device=accelerator.device, dtype=args.weight_dtype)
    num_frames = args.num_frames
    min_guidance_scale = 1.0
    num_inference_steps = args.num_inference_steps
    motion_bucket_id = 127

    pipeline._guidance_scale = max_guidance_scale
    original_pipeline._guidance_scale = max_guidance_scale

    unchanged_added_time_ids = torch.tensor(
        [fps, motion_bucket_id, noise_aug_strength],
        device=accelerator.device,
        dtype=args.weight_dtype,
    )

    device = accelerator.device

    result_videos = []
    original_result_videos = []
    ground_truth_videos = []

    if args.eval_metric_names is not None:
        result_numerical = defaultdict(list)
    for idx, batch in enumerate(
        tqdm(
            eval_dataloader,
            desc=f"Evaluation step {step+1}",
            disable=not accelerator.is_main_process,
        )
    ):
        for k, v in batch.items():
            batch[k] = v.to(accelerator.device, args.weight_dtype)
        batch_size = batch["clip"].shape[0]

        latents = denoising(
            pipeline=pipeline,
            batch=batch,
            accelerator=accelerator,
            args=args,
            idx=idx,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            unchanged_added_time_ids=unchanged_added_time_ids,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            noise_aug_strength=noise_aug_strength,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            timesteps=pipeline.scheduler.timesteps,
            latent=init_latent.detach().clone(),
            batch_size=batch_size,
        )
        # diffusion process ends
        latents /= model.vae.config.scaling_factor
        vid_tensor = model.vae.decode(latents.flatten(0, 1), num_frames=latents.shape[1]).sample.unsqueeze(0)
        frames = convert_video(vid_tensor)
        result_videos.append(frames)
        if args.eval_metric_names is not None:
            for name in args.eval_metric_names:
                result_numerical[name].append(
                    getattr(eval_metrics, name)(
                        (batch["vae"] + 1) / 2,
                        frames / 255.0,
                        device=accelerator.device,
                    )
                )
        # original_pipeline starts
        latents = denoising(
            pipeline=original_pipeline,
            batch=batch,
            accelerator=accelerator,
            args=args,
            idx=idx,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            unchanged_added_time_ids=unchanged_added_time_ids,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            noise_aug_strength=noise_aug_strength,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            timesteps=pipeline.scheduler.timesteps,
            latent=init_latent.detach().clone(),
            batch_size=batch_size,
        )
        # diffusion process ends
        latents /= model.vae.config.scaling_factor
        vid_tensor = model.vae.decode(latents.flatten(0, 1), num_frames=latents.shape[1]).sample.unsqueeze(0)
        frames = convert_video(vid_tensor)
        original_result_videos.append(frames)
        # gt videos
        ground_truth_videos.append(convert_video(batch["vae"]))
    # numerical evaluation
    if args.eval_metric_names is not None:
        for name in args.eval_metric_names:
            result_numerical[f"{name}"] = torch.stack(result_numerical[f"{name}"], dim=0)
    result_videos = torch.cat(result_videos, dim=0)
    original_result_videos = torch.cat(original_result_videos, dim=0)
    ground_truth_videos = torch.cat(ground_truth_videos, dim=0)
    # collect
    accelerator.wait_for_everyone()
    result_videos = gather_different_size_tensor(result_videos, accelerator)
    original_result_videos = gather_different_size_tensor(original_result_videos, accelerator)
    ground_truth_videos = gather_different_size_tensor(ground_truth_videos, accelerator)

    result_numerical_dict = {}
    if args.eval_metric_names is not None:
        for name in args.eval_metric_names:
            exp_result_numerical = gather_different_size_tensor(result_numerical[f"{name}"], accelerator)
            result_numerical_dict[f"eval/{name}"] = exp_result_numerical.mean().detach().cpu().item()
    # log
    if accelerator.is_main_process:
        total_size = result_videos.shape[0]
        match args.report_to:
            # case "tensorboard":
            #     for idx, video in enumerate(result_videos):
            #         if video.ndim != 5:
            #             video = video.unsqueeze(0)
            #         accelerator.get_tracker("tensorboard").writer.add_video(
            #             f"eval_video/{idx}",
            #             video,
            #             global_step=step,
            #             fps=1 / 100,
            #         )
            #     if args.eval_metric_names is not None:
            #         accelerator.log(
            #             result_numerical_dict,
            #             step=step,
            #         )
            case "wandb":
                for videos, name in zip(
                    [result_videos, original_result_videos, ground_truth_videos],
                    [
                        "eval_video",
                        "original_eval_video",
                        "ground_truth_video",
                    ],
                ):
                    for idx, video in enumerate(videos):
                        accelerator.get_tracker("wandb").log(
                            {
                                f"{name}/{str(idx).zfill(int(math.log10(total_size) + 1))}": wandb.Video(
                                    video.to(torch.float32).cpu(), fps=6
                                ),
                            }
                        )
                if args.eval_metric_names is not None:
                    accelerator.log(
                        result_numerical_dict,
                    )
            # case None:
            #     save_path = os.path.join(
            #         "inference_results",
            #         f"{args.exp_name}",
            #     )
            #     os.makedirs(save_path, exist_ok=True)
            #     for idx, video in enumerate(result_videos):
            #         video = video.permute(0, 2, 3, 1).cpu()
            #         torchvision.io.write_video(
            #             os.path.join(
            #                 save_path,
            #                 f"{str(idx).zfill(int(math.log10(total_size) + 1))}.mp4",
            #             ),
            #             video,
            #             fps=7,
            #             video_codec="h264",
            #             options={"crf": "10"},
            #         )
            case _:
                raise NotImplementedError
    del pipeline, original_pipeline
    torch.cuda.empty_cache()
    model.train()
