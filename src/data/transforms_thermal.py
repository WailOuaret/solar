from __future__ import annotations

from typing import Callable

try:
    from torchvision import transforms
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for thermal transforms") from exc


def build_thermal_transform(image_size: int, is_train: bool = True) -> Callable:
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    ops = [transforms.Resize((image_size, image_size))]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        )
    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)

