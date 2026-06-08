from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BENCHMARK_DIR = Path("data") / "benchmark_questions"


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


def _new_canvas(width: int = 1300, height: int = 760) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def _line_height(font: ImageFont.ImageFont, spacing: int = 12) -> int:
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent + spacing
    return 34 + spacing


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    spacing: int = 12,
) -> None:
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill="black", font=font)
        current_y += _line_height(font, spacing)


def _save_text_question(output_path: Path, lines: list[str]) -> None:
    image, draw = _new_canvas()
    _draw_lines(draw, lines, 70, 70, _load_font(40))
    image.save(output_path)


def create_q09_parabola_vertex(output_path: Path) -> None:
    _save_text_question(
        output_path,
        [
            "For f(x) = x^2 - 4x + 3, what is the vertex of the parabola?",
            "A) (1, 0)",
            "B) (2, -1)",
            "C) (3, 0)",
            "D) (4, 3)",
        ],
    )


def create_q10_derivative(output_path: Path) -> None:
    _save_text_question(
        output_path,
        [
            "If f(x) = 3x^2 - 5x + 2, what is f'(2)?",
            "A) 5",
            "B) 7",
            "C) 9",
            "D) 12",
        ],
    )


def create_q11_limit(output_path: Path) -> None:
    _save_text_question(
        output_path,
        [
            "lim x -> 2 of (x^2 - 4) / (x - 2) equals?",
            "A) 2",
            "B) 4",
            "C) 0",
            "D) undefined",
        ],
    )


def create_q12_integral(output_path: Path) -> None:
    _save_text_question(
        output_path,
        [
            "What is the integral of 2x dx?",
            "A) 2x",
            "B) x^2 + C",
            "C) x^2",
            "D) 2x^2 + C",
        ],
    )


def create_q13_geometry_angle(output_path: Path) -> None:
    image, draw = _new_canvas()
    title_font = _load_font(40)
    body_font = _load_font(34)

    triangle = [(230, 420), (520, 160), (810, 420)]
    draw.polygon(triangle, outline="black", fill="#f6fbff")
    draw.line(triangle + [triangle[0]], fill="black", width=4)
    draw.text((265, 370), "50 deg", fill="black", font=body_font)
    draw.text((690, 370), "60 deg", fill="black", font=body_font)
    draw.text((500, 205), "?", fill="black", font=_load_font(52))

    _draw_lines(
        draw,
        [
            "In a triangle, two angles are 50 degrees and 60 degrees.",
            "What is the third angle?",
            "A) 60 degrees",
            "B) 70 degrees",
            "C) 80 degrees",
            "D) 90 degrees",
        ],
        70,
        500,
        title_font,
        spacing=8,
    )
    image.save(output_path)


def _to_graph_point(origin: tuple[int, int], scale: int, x_value: float, y_value: float) -> tuple[int, int]:
    origin_x, origin_y = origin
    return int(origin_x + x_value * scale), int(origin_y - y_value * scale)


def create_q14_parabola_graph(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    origin = (420, 390)
    scale = 80

    draw.line((100, origin[1], 750, origin[1]), fill="black", width=3)
    draw.line((origin[0], 90, origin[0], 570), fill="black", width=3)
    draw.text((755, origin[1] - 20), "x", fill="black", font=font)
    draw.text((origin[0] + 12, 80), "y", fill="black", font=font)

    for x_tick in range(-3, 5):
        x_pos, y_pos = _to_graph_point(origin, scale, x_tick, 0)
        draw.line((x_pos, y_pos - 8, x_pos, y_pos + 8), fill="black", width=2)
        draw.text((x_pos - 8, y_pos + 12), str(x_tick), fill="black", font=_load_font(22))

    points = []
    for index in range(161):
        x_value = -1.0 + index * 0.03125
        y_value = (x_value - 1) * (x_value - 3)
        points.append(_to_graph_point(origin, scale, x_value, y_value))
    draw.line(points, fill="#1d5fbf", width=4)
    draw.ellipse((origin[0] + scale - 7, origin[1] - 7, origin[0] + scale + 7, origin[1] + 7), fill="#b00020")
    draw.ellipse((origin[0] + 3 * scale - 7, origin[1] - 7, origin[0] + 3 * scale + 7, origin[1] + 7), fill="#b00020")

    _draw_lines(
        draw,
        [
            "According to the graph, what are the approximate x-intercepts?",
            "A) 0 and 2",
            "B) 1 and 3",
            "C) -1 and 3",
            "D) 2 and 4",
        ],
        70,
        590,
        font,
        spacing=6,
    )
    image.save(output_path)


def create_q15_chart_reasoning(output_path: Path) -> None:
    image, draw = _new_canvas()
    title_font = _load_font(40)
    font = _load_font(32)
    draw.text((70, 50), "Subject Scores", fill="black", font=title_font)

    chart_left = 170
    chart_top = 140
    chart_bottom = 470
    bar_width = 120
    gap = 90
    scores = [("Math", 80, "#2f75b5"), ("Physics", 65, "#70ad47"), ("Chemistry", 75, "#ed7d31")]

    draw.line((chart_left - 40, chart_bottom, chart_left + 620, chart_bottom), fill="black", width=3)
    draw.line((chart_left - 40, chart_top, chart_left - 40, chart_bottom), fill="black", width=3)
    for index, (label, score, color) in enumerate(scores):
        x = chart_left + index * (bar_width + gap)
        bar_top = chart_bottom - int(score * 3.6)
        draw.rectangle((x, bar_top, x + bar_width, chart_bottom), fill=color, outline="black")
        draw.text((x + 20, bar_top - 38), str(score), fill="black", font=font)
        draw.text((x - 12, chart_bottom + 14), label, fill="black", font=font)

    _draw_lines(
        draw,
        [
            "Which subject has the highest score?",
            "A) Physics",
            "B) Math",
            "C) Chemistry",
            "D) Biology",
        ],
        70,
        540,
        font,
        spacing=8,
    )
    image.save(output_path)


def create_q16_mixed_math_visual(output_path: Path) -> None:
    image, draw = _new_canvas()
    title_font = _load_font(40)
    font = _load_font(34)

    draw.text((70, 55), "Rectangle Area", fill="black", font=title_font)
    rect = (150, 160, 660, 430)
    draw.rectangle(rect, outline="black", width=5, fill="#fff8e8")
    draw.text((365, 445), "width = x", fill="black", font=font)
    draw.text((690, 270), "height = 5", fill="black", font=font)
    draw.text((320, 265), "area = 40", fill="black", font=font)

    _draw_lines(
        draw,
        [
            "If the area is 40, what is x?",
            "A) 6",
            "B) 8",
            "C) 10",
            "D) 12",
        ],
        70,
        530,
        font,
        spacing=8,
    )
    image.save(output_path)


def create_all_benchmark_questions(output_dir: Path = BENCHMARK_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generators = [
        ("q09_parabola_vertex.png", create_q09_parabola_vertex),
        ("q10_derivative.png", create_q10_derivative),
        ("q11_limit.png", create_q11_limit),
        ("q12_integral.png", create_q12_integral),
        ("q13_geometry_angle.png", create_q13_geometry_angle),
        ("q14_parabola_graph.png", create_q14_parabola_graph),
        ("q15_chart_reasoning.png", create_q15_chart_reasoning),
        ("q16_mixed_math_visual.png", create_q16_mixed_math_visual),
    ]

    created_paths = []
    for file_name, generator in generators:
        output_path = output_dir / file_name
        generator(output_path)
        created_paths.append(output_path)
    return created_paths


if __name__ == "__main__":
    paths = create_all_benchmark_questions()
    for path in paths:
        print(f"Created benchmark question image at: {path}")
