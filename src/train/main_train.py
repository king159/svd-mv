import time

import accelerate
import torch
import yaml
from diffusers.training_utils import EMAModel
from tqdm import tqdm

from ..eval import eval_every_n_step, save_every_n_step
from .data import get_dataloader
from .i2v_model import I2VModel, I2VProcessor
from .lr_scheduler import get_lr_scheduler
from .optimizer import get_optimizer
from .params import parse_args
from .train_one_step import train_one_step


def train_model(config_path: str) -> None:
    with open(config_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=args.exp_name if args.report_to == "wandb" else "tensorboard",
    )
    accelerate.utils.set_seed(args.seed)
    args.world_size = accelerator.state.num_processes
    args.world_rank = accelerator.state.process_index
    accelerator.wait_for_everyone()
    args.exp_name = args.exp_name + time.strftime("_%m%d%H%M")
    # log
    if accelerator.is_main_process:
        if args.report_to == "tensorboard":
            temp = {}
            for k, v in vars(args).items():
                if isinstance(v, list):
                    temp[k] = str(v)
                else:
                    temp[k] = v
            accelerator.init_trackers(
                project_name=args.exp_name,
                config=temp,
            )
        elif args.report_to == "wandb":
            accelerator.init_trackers(
                project_name=args.project_name,
                config=vars(args),
                init_kwargs={"wandb": {"name": args.exp_name}},
            )
        args.weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        args.weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        args.weight_dtype = torch.bfloat16
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
    processor = I2VProcessor()
    model = I2VModel()
    if args.use_ema:
        from diffusers import UNet2DConditionModel

        ema_unet = EMAModel(
            model.unet.parameters(),
            model_cls=UNet2DConditionModel,
            decay=args.ema_decay,
            model_config=model.unet.config,
        )
        model.ema_unet = ema_unet
        model.ema_unet.to(accelerator.device, args.weight_dtype)
    model.to(accelerator.device, args.weight_dtype)
    if args.freeze_temporal:
        model.set_temporal_grad(False)
    if args.freeze_spatial:
        model.set_spatial_grad(False)
    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
    if accelerator.is_main_process:
        print(
            "Trainable parameters:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            args,
        )
    train_dataloader = get_dataloader(
        args=args,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        dataset_path=args.train_dataset_path,
        world_rank=args.world_rank,
        world_size=args.world_size,
        num_examples=args.num_examples,
        num_frames=args.num_frames,
        processor=processor,
    )
    eval_dataloader = get_dataloader(
        args=args,
        dataset_type=args.dataset_type,
        batch_size=args.eval_batch_size,
        dataset_path=args.eval_dataset_path,
        world_rank=args.world_rank,
        world_size=args.world_size,
        num_examples=args.eval_num_examples,
        num_frames=args.num_frames,
        processor=processor,
    )
    total_training_steps = train_dataloader.num_batches * args.num_epochs // args.gradient_accumulation_steps
    optimizer = get_optimizer(
        name=args.optimizer,
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.warmup_steps < 1:
        args.warmup_steps = int(total_training_steps * args.warmup_steps)
    else:
        args.warmup_steps = int(args.warmup_steps)
    lr_scheduler = get_lr_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
        num_training_steps=total_training_steps // args.gradient_accumulation_steps,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    eval_latent = torch.randn(
        1,
        args.num_frames,
        4,
        72,
        72,
        device=accelerator.device,
        dtype=args.weight_dtype,
    )
    global_step = 0
    # fps = torch.tensor(
    #     6, device=accelerator.device, dtype=args.weight_dtype, requires_grad=False
    # )
    # motion_bucket_id = torch.tensor(
    #     127, device=accelerator.device, dtype=torch.long, requires_grad=False
    # )
    # unchanged_added_time_ids = torch.stack([fps, motion_bucket_id])
    for epoch in range(0, args.num_epochs):
        progress_bar = tqdm(
            total=train_dataloader.num_batches,
            disable=not accelerator.is_main_process,
        )
        model.train()
        for batch in train_dataloader:
            progress_bar.update(1)
            progress_bar.set_description(
                time.strftime("%Y-%m-%d %H:%M:%S") + f" Train epoch {epoch+1}",
            )
            # train
            train_one_step(
                args=args,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                batch=batch,
                # unchanged_added_time_ids=unchanged_added_time_ids,
                step=global_step,
            )
            # save
            if (global_step + 1) % args.save_every_step == 0:
                save_every_n_step(
                    model=model,
                    args=args,
                    accelerator=accelerator,
                    step=global_step,
                    lr_scheduler=lr_scheduler,
                    optimizer=optimizer,
                )
                accelerator.wait_for_everyone()
            # eval
            if (global_step + 1) % args.eval_every_step == 0:
                if args.use_ema:
                    model.ema_unet.store(model.unet.parameters())
                    model.ema_unet.copy_to(model.unet.parameters())
                eval_every_n_step(
                    model=model,
                    step=global_step,
                    accelerator=accelerator,
                    args=args,
                    eval_dataloader=eval_dataloader,
                    latent=eval_latent,
                )
                if args.use_ema:
                    model.ema_unet.restore(model.unet.parameters())
                accelerator.wait_for_everyone()
            global_step += 1
