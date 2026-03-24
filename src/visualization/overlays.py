from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def draw_text_overlay(image_path: str | Path, text: str, output_path: str | Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, image.width - 10, 60), fill=(0, 0, 0))
    draw.text((20, 20), text, fill=(255, 255, 255))
    image.save(output_path)

