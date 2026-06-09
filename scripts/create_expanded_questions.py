from __future__ import annotations

import json
import math
import random
import textwrap
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont


EXPANDED_DIR = Path("data") / "expanded_questions"
GROUND_TRUTH_PATH = Path("data") / "expanded_ground_truth.json"


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


def _new_canvas(width: int = 1400, height: int = 850, color: str | tuple[int, int, int] = "white") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), color)
    return image, ImageDraw.Draw(image)


def _line_height(font: ImageFont.ImageFont, spacing: int = 10) -> int:
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent + spacing
    return 34 + spacing


def _wrap_line(line: str, width: int = 62) -> list[str]:
    if not line:
        return [""]
    if line.startswith(("A)", "B)", "C)", "D)", "E)")):
        return [line]
    return textwrap.wrap(line, width=width) or [line]


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int] = "black",
    spacing: int = 10,
    wrap_width: int = 62,
) -> int:
    current_y = y
    for line in lines:
        for wrapped in _wrap_line(line, wrap_width):
            draw.text((x, current_y), wrapped, fill=fill, font=font)
            current_y += _line_height(font, spacing)
    return current_y


def _question_lines(question: str, options: dict[str, str]) -> list[str]:
    return [question, ""] + [f"{letter}) {text}" for letter, text in options.items()]


def _save_text_question(output_path: Path, question: str, options: dict[str, str], title: str = "") -> None:
    image, draw = _new_canvas()
    title_font = _load_font(38)
    body_font = _load_font(36)
    y = 55
    if title:
        draw.text((70, y), title, fill="black", font=title_font)
        y += 70
    _draw_lines(draw, _question_lines(question, options), 70, y, body_font)
    image.save(output_path)


def _draw_table(draw: ImageDraw.ImageDraw, rows: list[list[str]], left: int, top: int, cell_w: int, cell_h: int, font: ImageFont.ImageFont) -> None:
    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        for col_index, value in enumerate(row):
            x = left + col_index * cell_w
            draw.rectangle((x, y, x + cell_w, y + cell_h), outline="black", width=3, fill="#fbfdff")
            draw.text((x + 18, y + 18), value, fill="black", font=font)


def _save_bar_chart(output_path: Path, title: str, values: list[tuple[str, int, str]], question: str, options: dict[str, str]) -> None:
    image, draw = _new_canvas()
    title_font = _load_font(40)
    body_font = _load_font(32)
    draw.text((70, 50), title, fill="black", font=title_font)

    chart_left = 160
    chart_bottom = 480
    bar_width = 115
    gap = 85
    draw.line((chart_left - 45, chart_bottom, chart_left + 720, chart_bottom), fill="black", width=3)
    draw.line((chart_left - 45, 140, chart_left - 45, chart_bottom), fill="black", width=3)
    for index, (label, value, color) in enumerate(values):
        x = chart_left + index * (bar_width + gap)
        bar_top = chart_bottom - value * 35
        draw.rectangle((x, bar_top, x + bar_width, chart_bottom), fill=color, outline="black")
        draw.text((x + 36, bar_top - 36), str(value), fill="black", font=body_font)
        draw.text((x - 10, chart_bottom + 16), label, fill="black", font=body_font)

    _draw_lines(draw, _question_lines(question, options), 70, 565, body_font, spacing=7)
    image.save(output_path)


def _save_line_chart(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), "Weekly Reading Minutes", fill="black", font=_load_font(40))
    left, top, bottom = 140, 140, 500
    points = [("Mon", 2), ("Tue", 4), ("Wed", 3), ("Thu", 6)]
    draw.line((left, bottom, 760, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)
    coords = []
    for index, (label, value) in enumerate(points):
        x = left + 120 + index * 150
        y = bottom - value * 45
        coords.append((x, y))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#1f77b4")
        draw.text((x - 24, bottom + 18), label, fill="black", font=font)
        draw.text((x - 8, y - 42), str(value), fill="black", font=font)
    draw.line(coords, fill="#1f77b4", width=4)
    _draw_lines(
        draw,
        _question_lines(
            "On which day were the reading minutes highest?",
            {"A": "Monday", "B": "Tuesday", "C": "Wednesday", "D": "Thursday", "E": "Same each day"},
        ),
        70,
        565,
        font,
        spacing=7,
    )
    image.save(output_path)


def _save_pie_chart(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), "Club Choices", fill="black", font=_load_font(40))
    box = (120, 150, 520, 550)
    slices = [("Art", 90, "#f4a261"), ("Music", 135, "#2a9d8f"), ("Sports", 90, "#e76f51"), ("Drama", 45, "#457b9d")]
    start = 0
    for label, angle, color in slices:
        draw.pieslice(box, start=start, end=start + angle, fill=color, outline="black")
        start += angle
    legend_x = 610
    legend_y = 180
    for index, (label, angle, color) in enumerate(slices):
        y = legend_y + index * 70
        draw.rectangle((legend_x, y, legend_x + 36, y + 36), fill=color, outline="black")
        draw.text((legend_x + 55, y - 2), f"{label}: {angle} deg", fill="black", font=font)
    _draw_lines(
        draw,
        _question_lines(
            "Which club has the largest slice in the chart?",
            {"A": "Art", "B": "Music", "C": "Sports", "D": "Drama", "E": "They are equal"},
        ),
        70,
        620,
        font,
        spacing=7,
    )
    image.save(output_path)


def _save_triangle(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), "Triangle Angles", fill="black", font=_load_font(40))
    triangle = [(220, 460), (560, 170), (900, 460)]
    draw.polygon(triangle, outline="black", fill="#eef7ff")
    draw.line(triangle + [triangle[0]], fill="black", width=5)
    draw.text((260, 405), "65 deg", fill="black", font=font)
    draw.text((760, 405), "45 deg", fill="black", font=font)
    draw.text((540, 220), "?", fill="black", font=_load_font(54))
    _draw_lines(
        draw,
        _question_lines(
            "Two angles of a triangle are 65 degrees and 45 degrees. What is the third angle?",
            {"A": "60 degrees", "B": "65 degrees", "C": "70 degrees", "D": "75 degrees", "E": "80 degrees"},
        ),
        70,
        555,
        font,
        spacing=7,
    )
    image.save(output_path)


def _save_circle(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), "Circle Center Angle", fill="black", font=_load_font(40))
    center = (430, 340)
    radius = 190
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline="black", width=5, fill="#fffdf2")
    draw.line((center[0], center[1], center[0] + radius, center[1]), fill="black", width=4)
    end_x = center[0] + int(radius * math.cos(math.radians(-80)))
    end_y = center[1] + int(radius * math.sin(math.radians(-80)))
    draw.line((center[0], center[1], end_x, end_y), fill="black", width=4)
    draw.arc((center[0] - 70, center[1] - 70, center[0] + 70, center[1] + 70), start=-80, end=0, fill="#b00020", width=4)
    draw.text((center[0] + 60, center[1] - 80), "80 deg", fill="black", font=font)
    _draw_lines(
        draw,
        _question_lines(
            "The central angle shown in the circle is 80 degrees. What is its measure?",
            {"A": "40 degrees", "B": "60 degrees", "C": "80 degrees", "D": "100 degrees", "E": "120 degrees"},
        ),
        70,
        610,
        font,
        spacing=7,
    )
    image.save(output_path)


def _save_rectangle(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), "Rectangle Area", fill="black", font=_load_font(40))
    rect = (160, 180, 700, 470)
    draw.rectangle(rect, outline="black", width=5, fill="#fff8e8")
    draw.text((350, 490), "width = 9", fill="black", font=font)
    draw.text((730, 300), "height = 4", fill="black", font=font)
    _draw_lines(
        draw,
        _question_lines(
            "What is the area of the rectangle?",
            {"A": "13", "B": "26", "C": "32", "D": "36", "E": "40"},
        ),
        70,
        590,
        font,
        spacing=7,
    )
    image.save(output_path)


def _save_coordinate(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(30)
    draw.text((70, 50), "Coordinate Plane", fill="black", font=_load_font(40))
    origin = (420, 410)
    scale = 70
    draw.line((120, origin[1], 780, origin[1]), fill="black", width=3)
    draw.line((origin[0], 120, origin[0], 640), fill="black", width=3)
    for value in range(-4, 6):
        x = origin[0] + value * scale
        draw.line((x, origin[1] - 7, x, origin[1] + 7), fill="black", width=2)
        if value:
            draw.text((x - 8, origin[1] + 14), str(value), fill="black", font=_load_font(20))
    for value in range(-3, 5):
        y = origin[1] - value * scale
        draw.line((origin[0] - 7, y, origin[0] + 7, y), fill="black", width=2)
    point = (origin[0] + 3 * scale, origin[1] - 2 * scale)
    draw.ellipse((point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9), fill="#b00020")
    draw.text((point[0] + 18, point[1] - 20), "P", fill="black", font=font)
    _draw_lines(
        draw,
        _question_lines(
            "Point P is shown on the grid. What are its coordinates?",
            {"A": "(2, 3)", "B": "(3, 2)", "C": "(-3, 2)", "D": "(3, -2)", "E": "(2, -3)"},
        ),
        70,
        675,
        font,
        spacing=4,
    )
    image.save(output_path)


def _save_mixed_visual(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), "Boxes of Pens", fill="black", font=_load_font(40))
    for index in range(3):
        x = 130 + index * 160
        draw.rectangle((x, 170, x + 120, 290), outline="black", width=4, fill="#e7f4ff")
        draw.text((x + 28, 215), "5", fill="black", font=_load_font(46))
    draw.text((660, 210), "+ 4 loose pens", fill="black", font=font)
    _draw_lines(
        draw,
        _question_lines(
            "There are 3 boxes with 5 pens each and 4 loose pens. How many pens are there?",
            {"A": "15", "B": "17", "C": "19", "D": "20", "E": "24"},
        ),
        70,
        430,
        font,
        spacing=8,
    )
    image.save(output_path)


def _save_visual_table_math(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), "Ticket Table", fill="black", font=_load_font(40))
    _draw_table(
        draw,
        [["Ticket", "Price"], ["Child", "4"], ["Student", "6"], ["Adult", "9"]],
        120,
        145,
        260,
        70,
        font,
    )
    _draw_lines(
        draw,
        _question_lines(
            "A student buys 2 student tickets. What is the total price?",
            {"A": "6", "B": "10", "C": "12", "D": "15", "E": "18"},
        ),
        70,
        500,
        font,
        spacing=8,
    )
    image.save(output_path)


def _save_noisy(output_path: Path, base_lines: list[str], seed: int) -> None:
    image, draw = _new_canvas(color=(248, 248, 248))
    _draw_lines(draw, base_lines, 70, 70, _load_font(36), fill=(25, 25, 25))
    image = image.rotate(1.2, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(246, 246, 246))
    random.seed(seed)
    pixels = image.load()
    width, height = image.size
    for _ in range(18000):
        x = random.randrange(width)
        y = random.randrange(height)
        r, g, b = pixels[x, y]
        delta = random.randint(-20, 20)
        pixels[x, y] = tuple(max(0, min(255, value + delta)) for value in (r, g, b))
    image.save(output_path)


def _save_low_contrast(output_path: Path) -> None:
    image, draw = _new_canvas(color=(232, 232, 232))
    font = _load_font(36)
    _draw_lines(
        draw,
        _question_lines(
            "A box has 6 blue cards and 4 red cards. How many cards are in the box?",
            {"A": "8", "B": "9", "C": "10", "D": "12", "E": "14"},
        ),
        70,
        80,
        font,
        fill=(92, 92, 92),
    )
    image.save(output_path)


def _text_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "q17",
            "filename": "q17_turkish_paragraph.png",
            "question_type": "turkish",
            "question": "Paragrafta anlatilan kisi her sabah kitap okur ve sonra kisa notlar yazar. Bu kisinin en belirgin aliskanligi hangisidir?",
            "options": {"A": "Spor yapmak", "B": "Kitap okumak", "C": "Resim cizmek", "D": "Sarki soylemek", "E": "Yolculuk yapmak"},
            "correct_answer": "B",
            "expected_reasoning": "The paragraph says the person reads books every morning.",
            "notes": "Original Turkish paragraph-style text question.",
        },
        {
            "id": "q18",
            "filename": "q18_turkish_sentence_meaning.png",
            "question_type": "turkish",
            "question": "'Zamanini iyi kullanan ogrenci islerini son gune birakmaz.' cumlesinde vurgulanan dusunce hangisidir?",
            "options": {"A": "Planli olmak", "B": "Hizli kosmak", "C": "Sessiz kalmak", "D": "Cok uyumak", "E": "Oyun oynamak"},
            "correct_answer": "A",
            "expected_reasoning": "The sentence emphasizes using time well and planning ahead.",
            "notes": "Sentence meaning question.",
        },
        {
            "id": "q19",
            "filename": "q19_turkish_inference.png",
            "question_type": "turkish",
            "question": "Elif yagmuru gorunce semsiyesini cantasina koydu. Bu durumdan hangi sonuc cikarilir?",
            "options": {"A": "Hava sicaktir", "B": "Elif hazirliklidir", "C": "Canta bostur", "D": "Yol kisadir", "E": "Elif acikmistir"},
            "correct_answer": "B",
            "expected_reasoning": "Taking an umbrella when seeing rain shows preparation.",
            "notes": "Inference question.",
        },
        {
            "id": "q20",
            "filename": "q20_history_text.png",
            "question_type": "social",
            "question": "A town council keeps written records of decisions so later generations can learn what happened. This most directly helps which field?",
            "options": {"A": "History", "B": "Geometry", "C": "Music", "D": "Astronomy", "E": "Chemistry"},
            "correct_answer": "A",
            "expected_reasoning": "Written records are evidence used in history.",
            "notes": "History text question.",
        },
        {
            "id": "q21",
            "filename": "q21_geography_text.png",
            "question_type": "social",
            "question": "A region receives little rain and has very sparse plants. Which climate description best fits this region?",
            "options": {"A": "Desert", "B": "Rain forest", "C": "Tundra", "D": "Oceanic", "E": "Mountain"},
            "correct_answer": "A",
            "expected_reasoning": "Little rain and sparse plants describe desert conditions.",
            "notes": "Geography text question.",
        },
        {
            "id": "q22",
            "filename": "q22_social_reasoning.png",
            "question_type": "social",
            "question": "A class votes before choosing a project topic. Which idea is being practiced?",
            "options": {"A": "Random choice", "B": "Participation", "C": "Isolation", "D": "Trade", "E": "Migration"},
            "correct_answer": "B",
            "expected_reasoning": "Voting is a form of participation.",
            "notes": "Social reasoning question.",
        },
        {
            "id": "q23",
            "filename": "q23_math_word_problem.png",
            "question_type": "math",
            "question": "Mina has 8 pencils and buys 7 more. How many pencils does she have now?",
            "options": {"A": "12", "B": "13", "C": "14", "D": "15", "E": "16"},
            "correct_answer": "D",
            "expected_reasoning": "8 + 7 = 15.",
            "notes": "Simple word problem.",
        },
        {
            "id": "q24",
            "filename": "q24_math_ratio.png",
            "question_type": "math",
            "question": "The ratio of red beads to blue beads is 2 to 3. If there are 6 red beads, how many blue beads are there?",
            "options": {"A": "6", "B": "7", "C": "8", "D": "9", "E": "12"},
            "correct_answer": "D",
            "expected_reasoning": "Multiplying the ratio by 3 gives 6 red and 9 blue.",
            "notes": "Ratio question.",
        },
        {
            "id": "q25",
            "filename": "q25_math_equation.png",
            "question_type": "equation",
            "question": "Solve for x: 3x - 5 = 16",
            "options": {"A": "5", "B": "6", "C": "7", "D": "8", "E": "9"},
            "correct_answer": "C",
            "expected_reasoning": "3x = 21, so x = 7.",
            "notes": "Linear equation.",
        },
        {
            "id": "q26",
            "filename": "q26_math_percentage.png",
            "question_type": "math",
            "question": "What is 25 percent of 80?",
            "options": {"A": "10", "B": "15", "C": "20", "D": "25", "E": "30"},
            "correct_answer": "C",
            "expected_reasoning": "One quarter of 80 is 20.",
            "notes": "Percentage question.",
        },
        {
            "id": "q27",
            "filename": "q27_function_value.png",
            "question_type": "algebra",
            "question": "If f(x) = 2x + 3, what is f(4)?",
            "options": {"A": "8", "B": "9", "C": "10", "D": "11", "E": "12"},
            "correct_answer": "D",
            "expected_reasoning": "f(4) = 2 * 4 + 3 = 11.",
            "notes": "Function value.",
        },
        {
            "id": "q28",
            "filename": "q28_limit_text.png",
            "question_type": "calculus",
            "question": "What is the limit of x + 3 as x approaches 2?",
            "options": {"A": "3", "B": "4", "C": "5", "D": "6", "E": "7"},
            "correct_answer": "C",
            "expected_reasoning": "Substitute x = 2 into x + 3 to get 5.",
            "notes": "Limit text question.",
        },
        {
            "id": "q29",
            "filename": "q29_derivative_text.png",
            "question_type": "calculus",
            "question": "If f(x) = x^2, what is f'(3)?",
            "options": {"A": "3", "B": "4", "C": "5", "D": "6", "E": "9"},
            "correct_answer": "D",
            "expected_reasoning": "f'(x) = 2x, so f'(3) = 6.",
            "notes": "Derivative text question.",
        },
        {
            "id": "q30",
            "filename": "q30_integral_text.png",
            "question_type": "calculus",
            "question": "What is the integral of 4 dx?",
            "options": {"A": "4x + C", "B": "x^4 + C", "C": "4 + C", "D": "2x + C", "E": "x + C"},
            "correct_answer": "A",
            "expected_reasoning": "The antiderivative of constant 4 is 4x + C.",
            "notes": "Integral text question.",
        },
        {
            "id": "q31",
            "filename": "q31_parabola_text.png",
            "question_type": "algebra",
            "question": "The parabola y = (x - 1)^2 + 2 has vertex at which point?",
            "options": {"A": "(1, 2)", "B": "(2, 1)", "C": "(-1, 2)", "D": "(1, -2)", "E": "(0, 2)"},
            "correct_answer": "A",
            "expected_reasoning": "Vertex form gives vertex (1, 2).",
            "notes": "Parabola text question.",
        },
        {
            "id": "q40",
            "filename": "q40_physics_force.png",
            "question_type": "science",
            "question": "A cart with mass 3 kg accelerates at 2 m/s^2. What is the net force?",
            "options": {"A": "4 N", "B": "5 N", "C": "6 N", "D": "8 N", "E": "10 N"},
            "correct_answer": "C",
            "expected_reasoning": "F = m * a = 3 * 2 = 6 N.",
            "notes": "Physics force question.",
        },
        {
            "id": "q41",
            "filename": "q41_physics_energy.png",
            "question_type": "science",
            "question": "A lamp changes electrical energy mostly into light and what other form of energy?",
            "options": {"A": "Sound", "B": "Heat", "C": "Motion", "D": "Magnetism", "E": "Chemical"},
            "correct_answer": "B",
            "expected_reasoning": "Lamps commonly produce light and heat.",
            "notes": "Physics energy question.",
        },
        {
            "id": "q42",
            "filename": "q42_chemistry_reaction.png",
            "question_type": "science",
            "question": "When vinegar and baking soda bubble together, what evidence suggests a gas forms?",
            "options": {"A": "Bubbles", "B": "Silence", "C": "A shadow", "D": "A magnet", "E": "A ruler"},
            "correct_answer": "A",
            "expected_reasoning": "Bubbles are evidence of gas formation.",
            "notes": "Chemistry reaction question.",
        },
        {
            "id": "q43",
            "filename": "q43_biology_cell.png",
            "question_type": "science",
            "question": "Which cell part controls many activities and contains genetic material?",
            "options": {"A": "Cell wall", "B": "Nucleus", "C": "Vacuole", "D": "Ribosome", "E": "Membrane"},
            "correct_answer": "B",
            "expected_reasoning": "The nucleus contains genetic material and controls cell activities.",
            "notes": "Biology cell question.",
        },
        {
            "id": "q48",
            "filename": "q48_multi_step_problem.png",
            "question_type": "math",
            "question": "A shop sells 3 notebooks for 5 each and 2 pens for 2 each. What is the total cost?",
            "options": {"A": "15", "B": "17", "C": "19", "D": "21", "E": "25"},
            "correct_answer": "C",
            "expected_reasoning": "3 * 5 = 15 and 2 * 2 = 4, so total is 19.",
            "notes": "Multi-step arithmetic.",
        },
        {
            "id": "q50",
            "filename": "q50_adaptive_ambiguous.png",
            "question_type": "mixed",
            "question": "The question mentions a small table and a short calculation. If the value in row B is 7 and we add 5, what is the result?",
            "options": {"A": "10", "B": "11", "C": "12", "D": "13", "E": "14"},
            "correct_answer": "C",
            "expected_reasoning": "Row B value 7 plus 5 is 12.",
            "notes": "Ambiguous text mentioning table and calculation.",
        },
    ]


def _visual_specs() -> list[dict[str, Any]]:
    return [
        {"id": "q32", "filename": "q32_triangle_angle.png", "question_type": "geometry", "correct_answer": "C", "expected_reasoning": "180 - 65 - 45 = 70.", "notes": "Triangle visual.", "draw": _save_triangle},
        {"id": "q33", "filename": "q33_circle_angle.png", "question_type": "geometry", "correct_answer": "C", "expected_reasoning": "The central angle is labeled 80 degrees.", "notes": "Circle visual.", "draw": _save_circle},
        {"id": "q34", "filename": "q34_rectangle_area.png", "question_type": "geometry", "correct_answer": "D", "expected_reasoning": "9 * 4 = 36.", "notes": "Rectangle area visual.", "draw": _save_rectangle},
        {"id": "q35", "filename": "q35_coordinate_geometry.png", "question_type": "geometry", "correct_answer": "B", "expected_reasoning": "Point P is at x = 3 and y = 2.", "notes": "Coordinate grid visual.", "draw": _save_coordinate},
        {
            "id": "q36",
            "filename": "q36_bar_chart_reasoning.png",
            "question_type": "chart",
            "correct_answer": "B",
            "expected_reasoning": "Team Blue has value 8, the highest bar.",
            "notes": "Bar chart visual.",
            "draw": lambda path: _save_bar_chart(
                path,
                "Team Points",
                [("Red", 5, "#d9534f"), ("Blue", 8, "#337ab7"), ("Green", 6, "#5cb85c")],
                "Which team has the highest number of points?",
                {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow", "E": "All equal"},
            ),
        },
        {"id": "q37", "filename": "q37_line_chart_reasoning.png", "question_type": "chart", "correct_answer": "D", "expected_reasoning": "Thursday has the highest value, 6.", "notes": "Line chart visual.", "draw": _save_line_chart},
        {
            "id": "q38",
            "filename": "q38_table_reasoning.png",
            "question_type": "table",
            "correct_answer": "D",
            "expected_reasoning": "The table shows B has 9.",
            "notes": "Table reading visual.",
            "draw": lambda path: _save_table_question(
                path,
                "Score Table",
                [["Name", "Score"], ["A", "4"], ["B", "9"], ["C", "6"]],
                "Which row has the score 9?",
                {"A": "Row A", "B": "Row C", "C": "No row", "D": "Row B", "E": "All rows"},
            ),
        },
        {"id": "q39", "filename": "q39_pie_chart_reasoning.png", "question_type": "chart", "correct_answer": "B", "expected_reasoning": "Music has the largest slice at 135 degrees.", "notes": "Pie chart visual.", "draw": _save_pie_chart},
        {"id": "q44", "filename": "q44_mixed_math_visual.png", "question_type": "mixed", "correct_answer": "C", "expected_reasoning": "3 boxes of 5 plus 4 loose pens gives 19.", "notes": "Mixed visual arithmetic.", "draw": _save_mixed_visual},
        {"id": "q49", "filename": "q49_visual_table_math.png", "question_type": "table", "correct_answer": "C", "expected_reasoning": "Two student tickets cost 2 * 6 = 12.", "notes": "Table plus arithmetic.", "draw": _save_visual_table_math},
    ]


def _save_table_question(output_path: Path, title: str, rows: list[list[str]], question: str, options: dict[str, str]) -> None:
    image, draw = _new_canvas()
    font = _load_font(32)
    draw.text((70, 50), title, fill="black", font=_load_font(40))
    _draw_table(draw, rows, 120, 145, 260, 70, font)
    _draw_lines(draw, _question_lines(question, options), 70, 500, font, spacing=8)
    image.save(output_path)


def _noise_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "q45",
            "filename": "q45_noisy_turkish.png",
            "question_type": "noisy_text",
            "correct_answer": "B",
            "expected_reasoning": "The text asks what Kerem reads every evening: a story.",
            "notes": "Noisy Turkish-style text.",
            "draw": lambda path: _save_noisy(
                path,
                _question_lines(
                    "Kerem her aksam kisa bir hikaye okur. Kerem ne okur?",
                    {"A": "Siir", "B": "Hikaye", "C": "Haber", "D": "Liste", "E": "Harita"},
                ),
                seed=45,
            ),
        },
        {
            "id": "q46",
            "filename": "q46_noisy_math.png",
            "question_type": "noisy_text",
            "correct_answer": "D",
            "expected_reasoning": "18 - 6 = 12.",
            "notes": "Noisy math text.",
            "draw": lambda path: _save_noisy(
                path,
                _question_lines(
                    "What is 18 - 6?",
                    {"A": "8", "B": "9", "C": "10", "D": "12", "E": "14"},
                ),
                seed=46,
            ),
        },
        {"id": "q47", "filename": "q47_low_contrast_text.png", "question_type": "low_contrast_text", "correct_answer": "C", "expected_reasoning": "6 + 4 = 10.", "notes": "Low contrast text image.", "draw": _save_low_contrast},
    ]


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "question_id": record["id"],
        "image_path": str(EXPANDED_DIR / record["filename"]).replace("\\", "/"),
        "question_type": record["question_type"],
        "difficulty": "expanded",
        "correct_answer": record["correct_answer"],
        "expected_reasoning": record["expected_reasoning"],
        "notes": record["notes"],
    }


def create_all_expanded_questions(output_dir: Path = EXPANDED_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    created_paths: list[Path] = []

    for spec in _text_specs():
        output_path = output_dir / spec["filename"]
        _save_text_question(output_path, spec["question"], spec["options"])
        records.append(_metadata(spec))
        created_paths.append(output_path)

    visual_records = _visual_specs() + _noise_specs()
    for spec in visual_records:
        output_path = output_dir / spec["filename"]
        draw_fn: Callable[[Path], None] = spec["draw"]
        draw_fn(output_path)
        records.append(_metadata(spec))
        created_paths.append(output_path)

    records.sort(key=lambda item: item["id"])
    GROUND_TRUTH_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return sorted(created_paths)


if __name__ == "__main__":
    paths = create_all_expanded_questions()
    for path in paths:
        print(f"Created expanded question image at: {path}")
    print(f"Created expanded ground truth at: {GROUND_TRUTH_PATH}")
