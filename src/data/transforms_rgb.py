from __future__ import annotations

from typing import Callable

try:
    from torchvision import transforms
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for RGB transforms") from exc


def build_rgb_transform(image_size: int, is_train: bool = True) -> Callable:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

