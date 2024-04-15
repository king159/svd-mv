import argparse
import sys
from argparse import Namespace


def parse_args(input_args: dict[str, str | int | float | bool]) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_rate",
        type=float,
        default=0.2,
        help="Rate of the classifier-free guidance during training.",
    )
    parser.add_argument(
        "--condition_p_mean",
        type=float,
        default=-3.0,
        help="Mean of the log normal noise distribution.",
    )
    parser.add_argument(
        "--condition_p_std",
        type=float,
        default=0.5,
        help="Standard deviation of the log normal noise distribution.",
    )
    parser.add_argument(
        "--freeze_spatial",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Freeze spatial layers of the unet.",
    )
    parser.add_argument(
        "--freeze_temporal",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Freeze temporal layers of the unet.",
    )
    parser.add_argument(
        "--use_ema",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use EMA model when training.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99999,
        help="EMA decay.",
    )
    parser.add_argument(
        "--multi_view",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Multi-view training.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of inference steps of svd.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=21,
        help="Number of frames to generate.",
    )
    parser.add_argument(
        "--p_std",
        type=float,
        default=1.6,
        help="Standard deviation of the log normal noise distribution.",
    )
    parser.add_argument(
        "--p_mean",
        type=float,
        default=1.0,
        help="Mean of the log normal noise distribution.",
    )
    parser.add_argument(
        "--eval_metric_names",
        nargs="+",
        type=str,
        help="List of evaluation metrics.",
    )
    parser.add_argument(
        "--i2v_config_path",
        type=str,
        default="sd2.1",
        help="Path to i2v config.",
    )
    parser.add_argument(
        "--lora",
        type=lambda x: eval(x),
        help="Use Lora.",
    )
    parser.add_argument(
        "--drop_last",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Drop last batch.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to train on.",
    )
    parser.add_argument(
        "--eval_num_examples",
        type=int,
        default=50,
        help="Number of examples to train on.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="Experiment name.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="SVD360",
        help="Project name.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a folder",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Path to load model from.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm.",
    )
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default="save",
        help="Path to save checkpoint.",
    )
    parser.add_argument(
        "--save_every_step",
        type=int,
        default=sys.maxsize,
        help="Save checkpoint every n step.",
    )
    parser.add_argument(
        "--eval_every_step",
        type=int,
        default=sys.maxsize,
        help="Evaluate every n step.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Report training metrics to wandb or tensorboard.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=float,
        default=0,
        help="Warmup steps.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=None,
        help="Path to file(s) with training data.",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=None,
        help="Path to file(s) with evaluation data.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="image_folder",
        help="Dataset type: image_folder.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Gradient checkpointing.",
    )
    args_list: list[str] = []
    for key, value in input_args.items():
        if isinstance(value, list):
            args_list.append(f"--{key}")
            for v in value:
                args_list.append(str(v))
        elif isinstance(value, bool):
            if value:
                args_list.append(f"--{key}")
            else:
                args_list.append(f"--no-{key}")
        else:
            args_list.append(f"--{key}")
            args_list.append(str(value))

    args = parser.parse_args(args_list)
    return args
