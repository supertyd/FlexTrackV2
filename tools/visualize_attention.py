"""Attention-map visualization for FlexTrack-V2.

Moved out of the model module so lib/models stays free of plotting deps
(cv2 / PIL / matplotlib) for pure-inference installs. Requires matplotlib + Pillow.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize_attention_map(img_path, attention_mask, out_path="attention.png", ratio=0.5, cmap="jet"):
    """Overlay a 2-D attention map on an image and save it.

    Args:
        img_path: path to the source image.
        attention_mask: attention tensor shaped [1, 1, H, W] (will be interpolated).
        out_path: where to write the overlay PNG.
        ratio: display scale factor.
        cmap: matplotlib colormap for the overlay.
    """
    img = Image.open(img_path, mode="r")
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    plt.imshow(img, alpha=1)
    plt.axis("off")

    mask = F.interpolate(attention_mask, size=(img_h, img_w), mode="bilinear", align_corners=False)
    normed_mask = (mask / mask.max() * 255).cpu().numpy()[0][0]
    plt.imshow(normed_mask, alpha=0.5, interpolation="bilinear", cmap=cmap)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
