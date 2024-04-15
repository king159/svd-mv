import functools
from typing import Optional

import torch
import torchvision.transforms.v2 as transforms
from diffusers import AutoencoderKLTemporalDecoder, EDMEulerScheduler
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    PretrainedConfig,
    PreTrainedModel,
)

from .my_svd_unet import UNetSpatioTemporalConditionModel


class I2VProcessor:
    def __init__(self) -> None:
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="feature_extractor",
        )
        self.clip_preprocess = functools.partial(self._clip_preprocess)
        self.vae_preprocess = functools.partial(self._vae_preprocess)

    def _vae_preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(size=[576, 576], antialias=True),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5],
                ),
            ]
        )(images)

    def _clip_preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        images = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5],
                ),
                transforms.Resize(size=[224, 224], antialias=True),
                transforms.Normalize(mean=[-1], std=[2.0]),
            ]
        )(images)
        # We un-normalize it after resizing.
        images = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values
        return images

    def __call__(self, batch_image: list[list[Image.Image]]) -> dict[str, torch.Tensor]:
        output_vae = []
        output_clip = []
        for frames in batch_image:
            output_vae.append(torch.stack(self.vae_preprocess(frames)).to(memory_format=torch.contiguous_format))
            output_clip.append(self.clip_preprocess(frames[0]).to(memory_format=torch.contiguous_format))
        return {
            "vae": torch.stack(output_vae),
            "clip": torch.cat(output_clip),
        }


class I2VModel(PreTrainedModel):
    def __init__(self) -> None:
        super().__init__(PretrainedConfig())
        for component, subfolder in zip(
            [
                AutoencoderKLTemporalDecoder,
                CLIPVisionModelWithProjection,
                UNetSpatioTemporalConditionModel,
            ],
            [
                "vae",
                "image_encoder",
                "unet",
            ],
        ):
            setattr(
                self,
                subfolder,
                component.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    subfolder=subfolder,
                    torch_dtype=torch.bfloat16,
                ),
            )
            if subfolder == "unet":
                unet_config = component.load_config(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    subfolder="unet",
                )
                # the original one is 256 * 3 = 768
                unet_config["projection_class_embeddings_input_dim"] = int(256 * 2)
                unet_config["pretrained_model_name_or_path"] = "stabilityai/stable-video-diffusion-img2vid-xt"
                setattr(
                    self,
                    subfolder,
                    component.from_pretrained(
                        **unet_config,
                        torch_dtype=torch.bfloat16,
                        subfolder=subfolder,
                        low_cpu_mem_usage=False,
                        ignore_mismatched_sizes=True,
                    ),
                )
            if subfolder != "unet":
                getattr(self, subfolder).requires_grad_(False)
        self.scheduler = EDMEulerScheduler(sigma_data=1, prediction_type="v_prediction")

    def to(self, device: torch.device, dtype: Optional[str] = None):
        self.unet.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)
        self.image_encoder.to(device, dtype=dtype)

    def set_temporal_grad(self, train: bool = True) -> None:
        for name, i in self.unet.named_parameters():
            if "temporal" in name:
                i.requires_grad_(train)

    def set_spatial_grad(self, train: bool = True) -> None:
        for name, i in self.unet.named_parameters():
            if "temporal" not in name:
                i.requires_grad_(train)
