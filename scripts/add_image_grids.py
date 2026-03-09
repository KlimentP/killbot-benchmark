import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def draw_grid_overlay(
    img: Image.Image,
    alpha: float = 0.35,
    line_width: int = 2,
    font_scale: float = 0.8,
    pad_frac: float = 0.06,
) -> Image.Image:
    alpha = clamp(alpha, 0.0, 1.0)
    width, height = img.size

    pad = int(round(min(width, height) * pad_frac))
    pad = max(pad, 20)
    out_width, out_height = width + 2 * pad, height + 2 * pad

    base = Image.new("RGBA", (out_width, out_height), (255, 255, 255, 255))
    base.paste(img.convert("RGBA"), (pad, pad))

    overlay = Image.new("RGBA", (out_width, out_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(10, int(round(min(width, height) * 0.035 * font_scale)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    x0, y0 = pad, pad
    x1, y1 = pad + width, pad + height
    ticks = list(range(0, 101, 10))

    grid_color = (0, 0, 0, int(round(255 * alpha)))
    axis_color = (0, 0, 0, int(round(255 * min(1.0, alpha + 0.25))))
    label_color = (0, 0, 0, 255)

    def x_pix(value: int) -> float:
        return x0 + (value / 100.0) * (x1 - x0)

    def y_pix(value: int) -> float:
        return y1 - (value / 100.0) * (y1 - y0)

    draw.rectangle([x0, y0, x1, y1], outline=axis_color, width=max(2, line_width))

    for tick in ticks:
        x_pos = x_pix(tick)
        y_pos = y_pix(tick)

        draw.line([(x_pos, y0), (x_pos, y1)], fill=grid_color, width=line_width)
        draw.line([(x0, y_pos), (x1, y_pos)], fill=grid_color, width=line_width)

        label = str(tick)
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]

        draw.text(
            (x_pos - label_width / 2, y1 + (pad - label_height) / 2),
            label,
            fill=label_color,
            font=font,
        )
        draw.text(
            (x0 - label_width - 6, y_pos - label_height / 2),
            label,
            fill=label_color,
            font=font,
        )

    x_label = "X (0-100)"
    y_label = "Y (0-100)"
    x_box = draw.textbbox((0, 0), x_label, font=font)
    y_box = draw.textbbox((0, 0), y_label, font=font)

    draw.text(
        (x0 + (x1 - x0) / 2 - (x_box[2] - x_box[0]) / 2, y1 + pad - (x_box[3] - x_box[1]) - 2),
        x_label,
        fill=label_color,
        font=font,
    )

    y_label_img = Image.new("RGBA", (y_box[2] - y_box[0] + 6, y_box[3] - y_box[1] + 6), (0, 0, 0, 0))
    y_draw = ImageDraw.Draw(y_label_img)
    y_draw.text((3, 3), y_label, fill=label_color, font=font)
    y_label_img = y_label_img.rotate(90, expand=True)
    base.alpha_composite(y_label_img, (2, y0 + (y1 - y0) // 2 - y_label_img.size[1] // 2))

    return Image.alpha_composite(base, overlay).convert("RGB")


def iter_source_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_output_path(source_path: Path, output_dir: Path) -> Path:
    return output_dir / source_path.name


def process_image(source_path: Path, output_path: Path, alpha: float, line_width: int, font_scale: float) -> None:
    with Image.open(source_path) as image:
        output_image = draw_grid_overlay(
            image,
            alpha=alpha,
            line_width=line_width,
            font_scale=font_scale,
        )
        output_image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add labeled 0-100 grid overlays to all images in fixtures/images/raw and save them to fixtures/images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("fixtures/images/raw"),
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/images"),
        help="Directory for processed images.",
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="Grid line opacity (0..1).")
    parser.add_argument("--linew", type=int, default=2, help="Grid line width in pixels.")
    parser.add_argument("--fontscale", type=float, default=0.8, help="Scale factor for label font size.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_images = iter_source_images(args.input_dir)
    if not source_images:
        raise SystemExit(f"No supported images found in {args.input_dir}")

    for source_path in source_images:
        output_path = build_output_path(source_path, args.output_dir)
        process_image(source_path, output_path, args.alpha, args.linew, args.fontscale)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
