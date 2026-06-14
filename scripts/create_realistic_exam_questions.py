from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REALISTIC_DIR = Path("data") / "realistic_exam_questions"
GROUND_TRUTH_PATH = Path("data") / "realistic_exam_ground_truth.json"


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


def _new_canvas(width: int = 1450, height: int = 930) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def _line_height(font: ImageFont.ImageFont, spacing: int = 9) -> int:
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent + spacing
    return 36 + spacing


def _wrap_line(line: str, width: int = 70) -> list[str]:
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
    wrap_width: int = 70,
    spacing: int = 9,
) -> int:
    current_y = y
    for line in lines:
        for wrapped in _wrap_line(line, wrap_width):
            draw.text((x, current_y), wrapped, fill="black", font=font)
            current_y += _line_height(font, spacing)
    return current_y


def _option_lines(options: dict[str, str]) -> list[str]:
    return [f"{letter}) {value}" for letter, value in options.items()]


def _question_lines(question: str, options: dict[str, str]) -> list[str]:
    return [question, ""] + _option_lines(options)


def _save_text_question(output_path: Path, title: str, question: str, options: dict[str, str]) -> None:
    image, draw = _new_canvas()
    draw.text((70, 45), title, fill="black", font=_load_font(40))
    _draw_lines(draw, _question_lines(question, options), 70, 125, _load_font(34), wrap_width=66)
    image.save(output_path)


def _draw_table(
    draw: ImageDraw.ImageDraw,
    rows: list[list[str]],
    left: int,
    top: int,
    cell_w: int,
    cell_h: int,
    font: ImageFont.ImageFont,
) -> None:
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            x = left + col_index * cell_w
            y = top + row_index * cell_h
            fill = "#edf5ff" if row_index == 0 else "#ffffff"
            draw.rectangle((x, y, x + cell_w, y + cell_h), outline="black", width=3, fill=fill)
            draw.text((x + 18, y + 17), value, fill="black", font=font)


def _save_science_chart(output_path: Path) -> None:
    image, draw = _new_canvas()
    title_font = _load_font(40)
    font = _load_font(30)
    draw.text((70, 45), "Science Data Interpretation", fill="black", font=title_font)
    draw.text((90, 115), "A student measures plant height after adding different amounts of fertilizer.", fill="black", font=font)

    left, bottom = 170, 520
    values = [("0 g", 8, "#8ecae6"), ("5 g", 12, "#219ebc"), ("10 g", 15, "#126782"), ("15 g", 14, "#023047")]
    draw.line((left - 50, bottom, left + 720, bottom), fill="black", width=3)
    draw.line((left - 50, 200, left - 50, bottom), fill="black", width=3)
    for index, (label, value, color) in enumerate(values):
        x = left + index * 170
        bar_top = bottom - value * 18
        draw.rectangle((x, bar_top, x + 95, bottom), fill=color, outline="black")
        draw.text((x + 24, bar_top - 34), str(value), fill="black", font=font)
        draw.text((x + 8, bottom + 15), label, fill="black", font=font)
    draw.text((80, 555), "Which fertilizer amount produced the greatest plant height?", fill="black", font=font)
    _draw_lines(
        draw,
        _option_lines({"A": "0 g", "B": "5 g", "C": "10 g", "D": "15 g", "E": "All equal"}),
        80,
        610,
        font,
    )
    image.save(output_path)


def _save_geometry_angle(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(31)
    draw.text((70, 45), "Geometry Angle Reasoning", fill="black", font=_load_font(40))
    draw.line((150, 260, 830, 260), fill="black", width=5)
    draw.line((150, 500, 830, 500), fill="black", width=5)
    draw.line((300, 600, 700, 170), fill="#b00020", width=5)
    draw.arc((520, 220, 650, 350), start=180, end=250, fill="#1d3557", width=4)
    draw.text((550, 300), "70 deg", fill="black", font=font)
    draw.arc((400, 420, 540, 560), start=0, end=110, fill="#1d3557", width=4)
    draw.text((430, 385), "x", fill="black", font=_load_font(42))
    _draw_lines(
        draw,
        _question_lines(
            "Two parallel lines are cut by a transversal. The marked interior angle is 70 degrees. What is x?",
            {"A": "55 degrees", "B": "70 degrees", "C": "90 degrees", "D": "110 degrees", "E": "130 degrees"},
        ),
        80,
        640,
        font,
        wrap_width=68,
    )
    image.save(output_path)


def _save_integral_area(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(30)
    draw.text((70, 45), "Integral and Area", fill="black", font=_load_font(40))
    origin = (190, 560)
    draw.line((origin[0], origin[1], 820, origin[1]), fill="black", width=3)
    draw.line((origin[0], 160, origin[0], origin[1]), fill="black", width=3)
    points = [(origin[0], origin[1] - 60), (760, origin[1] - 300)]
    draw.polygon([(origin[0], origin[1]), points[0], points[1], (760, origin[1])], fill="#d7ecff", outline="black")
    draw.line(points, fill="#1d3557", width=5)
    draw.text((390, 245), "y = 2x + 1", fill="black", font=font)
    draw.text((172, 575), "0", fill="black", font=font)
    draw.text((738, 575), "3", fill="black", font=font)
    _draw_lines(
        draw,
        _question_lines(
            "What is the area under y = 2x + 1 from x = 0 to x = 3?",
            {"A": "6", "B": "9", "C": "10", "D": "12", "E": "15"},
        ),
        80,
        660,
        font,
        wrap_width=68,
    )
    image.save(output_path)


def _save_parabola_graph(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(30)
    draw.text((70, 45), "Parabola Graph Interpretation", fill="black", font=_load_font(40))
    origin = (430, 410)
    scale = 70
    draw.line((120, origin[1], 850, origin[1]), fill="black", width=3)
    draw.line((origin[0], 120, origin[0], 650), fill="black", width=3)
    coords: list[tuple[int, int]] = []
    for step in range(-30, 71):
        x_value = step / 10
        y_value = (x_value - 2) ** 2 - 1
        x = origin[0] + int(x_value * scale)
        y = origin[1] - int(y_value * scale)
        if 120 <= x <= 850 and 120 <= y <= 650:
            coords.append((x, y))
    if len(coords) > 1:
        draw.line(coords, fill="#b00020", width=4)
    vertex = (origin[0] + 2 * scale, origin[1] + scale)
    draw.ellipse((vertex[0] - 7, vertex[1] - 7, vertex[0] + 7, vertex[1] + 7), fill="black")
    draw.text((vertex[0] + 15, vertex[1] - 12), "(2, -1)", fill="black", font=font)
    _draw_lines(
        draw,
        _question_lines(
            "The graph of a parabola has the shown vertex. Which point is the vertex?",
            {"A": "(1, -2)", "B": "(2, -1)", "C": "(-2, 1)", "D": "(0, -1)", "E": "(2, 1)"},
        ),
        80,
        680,
        font,
    )
    image.save(output_path)


def _save_mixed_table(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(31)
    draw.text((70, 45), "Mixed Table Reasoning", fill="black", font=_load_font(40))
    _draw_table(
        draw,
        [["Ticket", "Price"], ["Student", "50"], ["Adult", "90"], ["Teacher", "70"]],
        110,
        135,
        275,
        75,
        font,
    )
    _draw_lines(
        draw,
        _question_lines(
            "A group buys 2 adult tickets and 3 student tickets. What is the total cost?",
            {"A": "230", "B": "280", "C": "330", "D": "360", "E": "410"},
        ),
        80,
        520,
        font,
    )
    image.save(output_path)


TEXT_SPECS: list[dict[str, Any]] = [
    {
        "id": "r01",
        "filename": "r01_lgs_math_word_problem.png",
        "question_type": "lgs_math",
        "title": "LGS-Style Math Word Problem",
        "question": "A class prepares gift boxes. Each box contains 3 pencils and 2 erasers. If 18 pencils are used, how many erasers are needed for the same number of boxes?",
        "options": {"A": "6", "B": "8", "C": "10", "D": "12", "E": "18"},
        "correct_answer": "D",
        "expected_reasoning": "18 pencils make 6 boxes; each box needs 2 erasers, so 12 erasers are needed.",
        "notes": "Original synthetic LGS-style proportional reasoning.",
    },
    {
        "id": "r02",
        "filename": "r02_lgs_turkish_inference.png",
        "question_type": "lgs_turkish",
        "title": "LGS-Style Turkish Paragraph",
        "question": "Mert, yarismadan once uzun sure calisti. Son gun yeni konu ogrenmek yerine daha once tuttugu notlari tekrar etti. Paragrafa gore Mert icin hangisi soylenebilir?",
        "options": {"A": "Plansiz davranmistir", "B": "Hazirligini pekistirmistir", "C": "Yarismadan vazgecmistir", "D": "Not tutmayi gereksiz bulmustur", "E": "Sadece sansina guvenmistir"},
        "correct_answer": "B",
        "expected_reasoning": "Repeating previous notes shows that Mert reinforces his preparation.",
        "notes": "Original Turkish inference item.",
    },
    {
        "id": "r05",
        "filename": "r05_yks_derivative.png",
        "question_type": "yks_calculus",
        "title": "YKS-Style Derivative",
        "question": "For f(x) = x^3 - 4x, what is f'(2)?",
        "options": {"A": "2", "B": "4", "C": "8", "D": "10", "E": "12"},
        "correct_answer": "C",
        "expected_reasoning": "f'(x) = 3x^2 - 4, so f'(2) = 8.",
        "notes": "Original synthetic derivative question.",
    },
    {
        "id": "r08",
        "filename": "r08_yks_probability.png",
        "question_type": "yks_probability",
        "title": "YKS-Style Probability",
        "question": "A box contains 3 red, 2 blue, and 1 green ball. One ball is chosen at random. What is the probability that the chosen ball is not green?",
        "options": {"A": "1/6", "B": "1/3", "C": "1/2", "D": "2/3", "E": "5/6"},
        "correct_answer": "E",
        "expected_reasoning": "There are 5 non-green balls out of 6 total balls.",
        "notes": "Original synthetic probability question.",
    },
    {
        "id": "r09",
        "filename": "r09_yks_physics_work.png",
        "question_type": "yks_physics",
        "title": "YKS-Style Physics",
        "question": "A constant horizontal force of 12 N moves a box 5 m in the direction of the force. How much work is done?",
        "options": {"A": "17 J", "B": "30 J", "C": "60 J", "D": "72 J", "E": "120 J"},
        "correct_answer": "C",
        "expected_reasoning": "Work is force times displacement, so 12 * 5 = 60 J.",
        "notes": "Original synthetic work-energy question.",
    },
    {
        "id": "r10",
        "filename": "r10_yks_chemistry_mole.png",
        "question_type": "yks_chemistry",
        "title": "YKS-Style Chemistry",
        "question": "In the reaction 2H2 + O2 -> 2H2O, how many moles of O2 are required to react completely with 4 moles of H2?",
        "options": {"A": "1 mol", "B": "2 mol", "C": "3 mol", "D": "4 mol", "E": "8 mol"},
        "correct_answer": "B",
        "expected_reasoning": "The H2:O2 mole ratio is 2:1, so 4 mol H2 needs 2 mol O2.",
        "notes": "Original synthetic stoichiometry question.",
    },
    {
        "id": "r11",
        "filename": "r11_yks_biology_genetics.png",
        "question_type": "yks_biology",
        "title": "YKS-Style Biology",
        "question": "Two heterozygous pea plants, Aa and Aa, are crossed. What is the probability of an offspring with genotype aa?",
        "options": {"A": "1/4", "B": "1/3", "C": "1/2", "D": "3/4", "E": "1"},
        "correct_answer": "A",
        "expected_reasoning": "Aa x Aa gives AA, Aa, Aa, aa, so aa has probability 1/4.",
        "notes": "Original synthetic genetics question.",
    },
]


VISUAL_SPECS: list[dict[str, Any]] = [
    {
        "id": "r03",
        "filename": "r03_lgs_science_chart_reasoning.png",
        "question_type": "lgs_science_chart",
        "correct_answer": "C",
        "expected_reasoning": "The 10 g bar has the greatest value, 15.",
        "notes": "Original synthetic science chart.",
        "draw": _save_science_chart,
    },
    {
        "id": "r04",
        "filename": "r04_lgs_geometry_angle.png",
        "question_type": "lgs_geometry",
        "correct_answer": "D",
        "expected_reasoning": "Same-side interior angles are supplementary, so x = 110 degrees.",
        "notes": "Original synthetic geometry visual.",
        "draw": _save_geometry_angle,
    },
    {
        "id": "r06",
        "filename": "r06_yks_integral_area.png",
        "question_type": "yks_calculus_visual",
        "correct_answer": "D",
        "expected_reasoning": "Integral from 0 to 3 of 2x + 1 is x^2 + x from 0 to 3, which is 12.",
        "notes": "Original synthetic integral-area visual.",
        "draw": _save_integral_area,
    },
    {
        "id": "r07",
        "filename": "r07_yks_parabola_graph.png",
        "question_type": "yks_graph",
        "correct_answer": "B",
        "expected_reasoning": "The graph labels the vertex as (2, -1).",
        "notes": "Original synthetic parabola graph.",
        "draw": _save_parabola_graph,
    },
    {
        "id": "r12",
        "filename": "r12_mixed_table_reasoning.png",
        "question_type": "mixed_table",
        "correct_answer": "C",
        "expected_reasoning": "2 adult tickets cost 180 and 3 student tickets cost 150, total 330.",
        "notes": "Original synthetic table reasoning.",
        "draw": _save_mixed_table,
    },
]


def _metadata(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "question_id": spec["id"],
        "image_path": str(REALISTIC_DIR / spec["filename"]).replace("\\", "/"),
        "question_type": spec["question_type"],
        "difficulty": "realistic_exam_synthetic",
        "correct_answer": spec["correct_answer"],
        "expected_reasoning": spec["expected_reasoning"],
        "notes": spec["notes"],
    }


def create_all_realistic_exam_questions(output_dir: Path = REALISTIC_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    records: list[dict[str, Any]] = []

    for spec in TEXT_SPECS:
        output_path = output_dir / spec["filename"]
        _save_text_question(output_path, spec["title"], spec["question"], spec["options"])
        created_paths.append(output_path)
        records.append(_metadata(spec))

    for spec in VISUAL_SPECS:
        output_path = output_dir / spec["filename"]
        spec["draw"](output_path)
        created_paths.append(output_path)
        records.append(_metadata(spec))

    records.sort(key=lambda item: item["id"])
    GROUND_TRUTH_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return sorted(created_paths)


if __name__ == "__main__":
    paths = create_all_realistic_exam_questions()
    for path in paths:
        print(f"Created realistic exam question image at: {path}")
    print(f"Created realistic exam ground truth at: {GROUND_TRUTH_PATH}")
