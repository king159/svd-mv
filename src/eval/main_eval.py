import time

import accelerate
import safetensors
import torch
import yaml

from ..eval import eval_every_n_step
from ..train.data import get_dataloader
from ..train.i2v_model import I2VModel, I2VProcessor
from ..train.params import parse_args


def eval_model(config_path: str) -> None:
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
    safetensors.torch.load_model(
        model,
        args.load_model_path,
        strict=False,
    )
    model.to(accelerator.device, args.weight_dtype)

    eval_dataloader = get_dataloader(
        args=args,
        dataset_type=args.dataset_type,
        batch_size=args.eval_batch_size,
        dataset_path=args.eval_dataset_path,
        world_rank=args.world_rank,
        world_size=args.world_size,
        num_examples=args.num_examples,
        num_frames=args.num_frames,
        processor=processor,
    )

    model = accelerator.prepare(model)
    eval_latent = torch.randn(
        1,
        args.num_frames,
        4,
        72,
        72,
        device=accelerator.device,
        dtype=args.weight_dtype,
    )

    eval_every_n_step(
        model=model,
        step=0,
        accelerator=accelerator,
        args=args,
        eval_dataloader=eval_dataloader,
        latent=eval_latent,
    )
