import lpips
import torch
import torch.nn.functional as F
from transformers import CLIPModel


@torch.no_grad()
def psnr_score(
    gt_images: torch.Tensor,
    pred_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    peak signal-to-noise ratio (PSNR) on one image
    """
    psnr_results = []
    gt_images = gt_images.to(device)
    pred_images = pred_images.to(device)
    for gt_image, pred_image in zip(gt_images, pred_images):
        mse = torch.mean((gt_image.float() - pred_image.float()) ** 2)
        psnr = -10 * torch.log10(mse)
        psnr_results.append(psnr)
    return torch.stack(psnr_results).mean()


@torch.no_grad()
def clip_score(
    gt_images: list[torch.Tensor],
    pred_images: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    clip image similarity score on one image
    """
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K").eval().to(device)
    gt_images = gt_images.to(device)
    pred_images = pred_images.to(device)
    clip_results = []
    for gt_image, pred_image in zip(gt_images, pred_images):
        gt_image = F.interpolate(gt_image, size=(224, 224), mode="bicubic")
        pred_image = F.interpolate(pred_image, size=(224, 224), mode="bicubic")
        img_gt_score = model.get_image_features(gt_image)
        img_gt_score /= img_gt_score.norm(dim=-1, keepdim=True)
        img_pr_score = model.get_image_features(pred_image)
        img_pr_score /= img_pr_score.norm(dim=-1, keepdim=True)
        score = (img_gt_score * img_pr_score).sum(dim=-1).mean()
        clip_results.append(score)
    del model
    return torch.stack(clip_results).mean()


@torch.no_grad()
def lpips_score(
    gt_images: list[torch.Tensor],
    pred_images: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    lpips_results = []
    model = lpips.LPIPS(net="vgg", verbose=False).to(device)
    gt_images = gt_images.to(device)
    pred_images = pred_images.to(device)
    for gt_image, pred_image in zip(gt_images, pred_images):
        lpips_results.append(model(gt_image, pred_image).mean())
    del model
    return torch.stack(lpips_results).mean()
