import functools
import io
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


try:
    # for slurm
    from petrel_client.client import Client

    client = Client(enable_mc=True)

except ImportError:
    pass


def mask2bounding_box(mask: np.array) -> tuple[int, int, int, int]:
    y, x, _ = np.nonzero(mask)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    return min_x, min_y, max_x, max_y


def custom_collate_fn(
    batch: list[dict[str, float | list[Image.Image]]],
    processor: Callable,
    multi_view: bool,
) -> dict[str, torch.Tensor]:
    images = processor([example["image"] for example in batch])
    if multi_view:
        elevation = torch.stack([torch.tensor(example["elevation"]) for example in batch])
        return {"elevation": elevation, **images}
    else:
        return images


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        world_rank: int = 0,
        world_size: int = 1,
        num_examples: Optional[int] = None,
        num_frames: int = 21,
    ) -> None:
        super(ImageFolderDataset).__init__()
        if "\n" in dataset_path:
            self.image_folder_path, self.meta_file = dataset_path.split("\n")
        else:
            self.image_folder_path = ""
            self.meta_file = dataset_path
        self.anno = []
        self.num_frames = num_frames
        with open(self.meta_file, "r") as f:
            anno = f.readlines()[:num_examples]
            self.anno = anno[world_rank::world_size]
            if len(self.anno) == 0:
                self.anno = [anno[0]]
        random.shuffle(self.anno)

    def __len__(self) -> int:
        return len(self.anno)

    def __getitem__(self, idx: int) -> dict[str, list[np.array]]:
        while True:
            this_anno = eval(self.anno[idx])
            image_paths = this_anno["image_path"]
            image_output = []
            try:
                if "s3://" in image_paths[0] or "s3://" in self.image_folder_path:
                    # roll_idx = random.randint(0, len(image_paths) - 1)
                    roll_idx = 0
                    elevation = this_anno["elevation"]
                    elevation = elevation[roll_idx:] + elevation[:roll_idx]
                    image_paths = image_paths[roll_idx:] + image_paths[:roll_idx]
                    image_paths = image_paths[: self.num_frames]
                    elevation = elevation[: self.num_frames]
                    min_x1, min_y1, max_x2, max_y2 = 10000, 10000, -10000, -10000
                    for i in image_paths:
                        image_path = os.path.join(self.image_folder_path, i)
                        image = np.array(Image.open(io.BytesIO(client.get(image_path))))
                        front_ground = image[:, :, :3]
                        back_ground = np.repeat(image[:, :, 3:4], 3, axis=-1)
                        front_ground[back_ground == 0] = 255
                        x1, y1, x2, y2 = mask2bounding_box(back_ground)
                        if x1 < min_x1:
                            min_x1 = x1
                        if y1 < min_y1:
                            min_y1 = y1
                        if x2 > max_x2:
                            max_x2 = x2
                        if y2 > max_y2:
                            max_y2 = y2
                        image_output.append(front_ground)
                    side_length = max(max_x2 - min_x1, max_y2 - min_y1)
                    center_x, center_y = (min_x1 + max_x2) // 2, (min_y1 + max_y2) // 2
                    min_x, max_x = (
                        max(0, center_x - side_length // 2 - 1),
                        center_x + side_length // 2 + 1,
                    )
                    min_y, max_y = (
                        max(0, center_y - side_length // 2 - 1),
                        center_y + side_length // 2 + 1,
                    )
                    crop_image_output = []
                    for image in image_output:
                        crop_image_output.append(image[min_y:max_y, min_x:max_x])
                    image_output = crop_image_output
                else:
                    roll_idx = random.randint(0, len(image_paths) - 1)
                    elevation = this_anno["elevation"]
                    elevation = elevation[roll_idx:] + elevation[:roll_idx]
                    image_paths = image_paths[roll_idx:] + image_paths[:roll_idx]
                    image_paths = image_paths[: self.num_frames]
                    elevation = elevation[: self.num_frames]
                    for i in image_paths:
                        image_path = os.path.join(self.image_folder_path, i)
                        image_output.append(np.array(Image.open(image_path))[:, :, :3])
                break
            except Exception as e:
                print(f"{e}: {image_path}")
                idx = random.randint(0, len(self.anno) - 1)
                continue
        return {
            "image": image_output,
            "elevation": elevation,
        }


def get_image_folder_dataset(
    args,
    dataset_path: str,
    world_rank: int = 0,
    world_size: int = 1,
    num_examples: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    drop_last: bool = True,
    processor: Callable = None,
    num_frames: int = 21,
) -> DataLoader:
    dataset = ImageFolderDataset(
        dataset_path=dataset_path,
        world_rank=world_rank,
        world_size=world_size,
        num_examples=num_examples,
        num_frames=num_frames,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=8,
        collate_fn=functools.partial(
            custom_collate_fn,
            processor=processor,
            multi_view=args.multi_view,
        ),
    )
    dataloader.num_batches = len(dataset) // batch_size
    return dataloader


def get_dataloader(
    args,
    dataset_type: str,
    dataset_path: str,
    world_rank: int = 0,
    world_size: int = 1,
    num_examples: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    num_frames: int = 21,
    processor: Callable = None,
) -> DataLoader:
    match dataset_type:
        case "image_folder":
            return get_image_folder_dataset(
                args,
                dataset_path=dataset_path,
                world_rank=world_rank,
                world_size=world_size,
                num_examples=num_examples,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                processor=processor,
                num_frames=num_frames,
            )
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
