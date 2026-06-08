from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFont


SAMPLE_DIR = Path("data") / "sample_questions"


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except Exception:
                pass
    return ImageFont.load_default()


def _line_height(font: ImageFont.ImageFont, spacing: int = 14) -> int:
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent + spacing
    return 32 + spacing


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int] = "black",
    spacing: int = 14,
) -> None:
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=fill, font=font)
        current_y += _line_height(font, spacing)


def _new_canvas(width: int = 1200, height: int = 620) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), color="white")
    return image, ImageDraw.Draw(image)


def _save_text_question(output_path: Path, lines: list[str]) -> None:
    image, draw = _new_canvas()
    _draw_lines(draw, lines, 60, 60, _load_font(46))
    image.save(output_path)


def _draw_star(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    center_y: int,
    outer_radius: int,
    inner_radius: int,
    fill: str,
) -> None:
    points = []
    for index in range(10):
        angle = math.radians(-90 + index * 36)
        radius = outer_radius if index % 2 == 0 else inner_radius
        points.append(
            (
                center_x + math.cos(angle) * radius,
                center_y + math.sin(angle) * radius,
            )
        )
    draw.polygon(points, fill=fill, outline="black")


def create_q01_text(output_path: Path) -> None:
    _save_text_question(
        output_path,
        [
            "What is 2 + 2?",
            "A) 3",
            "B) 4",
            "C) 5",
        ],
    )


def create_q02_math(output_path: Path) -> None:
    _save_text_question(
        output_path,
        [
            "What is 12 / 3 + 2?",
            "A) 4",
            "B) 6",
            "C) 8",
        ],
    )


def create_q03_equation(output_path: Path) -> None:
    _save_text_question(
        output_path,
        [
            "Solve for x: 2x + 3 = 11",
            "A) 3",
            "B) 4",
            "C) 5",
        ],
    )


def create_q04_table(output_path: Path) -> None:
    image, draw = _new_canvas(1200, 700)
    title_font = _load_font(42)
    body_font = _load_font(36)

    draw.text((60, 50), "Product Price Table", fill="black", font=title_font)

    left = 80
    top = 130
    cell_w = 300
    cell_h = 70
    rows = [
        ("Product", "Price"),
        ("Pen", "5"),
        ("Notebook", "20"),
        ("Bag", "50"),
    ]

    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        for col_index, value in enumerate(row):
            x = left + col_index * cell_w
            draw.rectangle((x, y, x + cell_w, y + cell_h), outline="black", width=3)
            draw.text((x + 20, y + 16), value, fill="black", font=body_font)

    _draw_lines(
        draw,
        [
            "Question: Which product costs 20?",
            "A) Pen",
            "B) Notebook",
            "C) Bag",
        ],
        60,
        450,
        body_font,
    )
    image.save(output_path)


def create_q05_chart(output_path: Path) -> None:
    image, draw = _new_canvas(1200, 720)
    title_font = _load_font(42)
    body_font = _load_font(34)

    draw.text((60, 45), "Fruit Values", fill="black", font=title_font)
    labels = [("Apples", 3, "#4f8f3a"), ("Bananas", 5, "#d6a51f"), ("Oranges", 2, "#d97030")]
    start_x = 90
    start_y = 150
    bar_unit = 80
    row_gap = 90

    for index, (label, value, color) in enumerate(labels):
        y = start_y + index * row_gap
        draw.text((start_x, y + 8), f"{label}:", fill="black", font=body_font)
        for bar_index in range(value):
            x = start_x + 210 + bar_index * (bar_unit + 10)
            draw.rectangle((x, y, x + bar_unit, y + 48), fill=color, outline="black")
        draw.text((start_x + 210 + value * 90 + 15, y + 8), str(value), fill="black", font=body_font)

    _draw_lines(
        draw,
        [
            "Question: Which fruit has the highest value?",
            "A) Apples",
            "B) Bananas",
            "C) Oranges",
        ],
        60,
        445,
        body_font,
    )
    image.save(output_path)


def create_q06_geometry(output_path: Path) -> None:
    image, draw = _new_canvas(1200, 700)
    title_font = _load_font(42)
    body_font = _load_font(34)

    draw.text((60, 45), "Rectangle Area", fill="black", font=title_font)
    rect = (130, 160, 610, 520)
    draw.rectangle(rect, outline="black", width=5, fill="#eef7ff")
    draw.text((305, 535), "width = 4", fill="black", font=body_font)
    draw.text((640, 315), "height = 3", fill="black", font=body_font)

    _draw_lines(
        draw,
        [
            "Question: What is the area?",
            "A) 7",
            "B) 12",
            "C) 14",
        ],
        720,
        170,
        body_font,
    )
    image.save(output_path)


def create_q07_mixed(output_path: Path) -> None:
    image, draw = _new_canvas(1200, 700)
    title_font = _load_font(42)
    body_font = _load_font(34)

    draw.text((60, 45), "Counting Stars", fill="black", font=title_font)
    draw.text((60, 120), "4 stars and 2 more stars", fill="black", font=body_font)

    star_positions = [(120, 245), (230, 245), (340, 245), (450, 245), (640, 245), (750, 245)]
    for x, y in star_positions:
        _draw_star(draw, x, y, 42, 18, "#ffd34d")

    draw.text((535, 225), "+", fill="black", font=_load_font(52))

    _draw_lines(
        draw,
        [
            "Question: How many stars are there in total?",
            "A) 5",
            "B) 6",
            "C) 8",
        ],
        60,
        370,
        body_font,
    )
    image.save(output_path)


def create_q08_noisy(output_path: Path) -> None:
    base_path = output_path.parent / "q01_text.png"
    if not base_path.exists():
        create_q01_text(base_path)

    random.seed(8)
    image = Image.open(base_path).convert("RGB")
    image = image.point(lambda value: int(235 + (value - 235) * 0.72))
    image = image.rotate(1.6, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(245, 245, 245))

    pixels = image.load()
    width, height = image.size
    for _ in range(14000):
        x = random.randrange(width)
        y = random.randrange(height)
        r, g, b = pixels[x, y]
        delta = random.randint(-18, 18)
        pixels[x, y] = (
            max(0, min(255, r + delta)),
            max(0, min(255, g + delta)),
            max(0, min(255, b + delta)),
        )

    image.save(output_path)


def create_all_sample_questions(output_dir: Path = SAMPLE_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generators = [
        ("q01_text.png", create_q01_text),
        ("q02_math.png", create_q02_math),
        ("q03_equation.png", create_q03_equation),
        ("q04_table.png", create_q04_table),
        ("q05_chart.png", create_q05_chart),
        ("q06_geometry.png", create_q06_geometry),
        ("q07_mixed.png", create_q07_mixed),
        ("q08_noisy.png", create_q08_noisy),
    ]

    created_paths = []
    for file_name, generator in generators:
        output_path = output_dir / file_name
        generator(output_path)
        created_paths.append(output_path)
    return created_paths


def create_sample_question_image(output_path: Path) -> None:
    create_q01_text(output_path)


if __name__ == "__main__":
    paths = create_all_sample_questions()
    for path in paths:
        print(f"Created sample question image at: {path}")
