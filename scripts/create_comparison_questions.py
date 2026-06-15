import json
import textwrap
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont


COMPARISON_DIR = Path("data") / "comparison_questions"
METADATA_PATH = COMPARISON_DIR / "comparison_questions.json"


QUESTION_RECORDS: list[dict[str, Any]] = [
    {
        "question_id": "cmp_m01",
        "subject": "matematik",
        "topic": "fonksiyon",
        "difficulty": "orta",
        "correct_answer": "C",
        "final_answer": "7",
        "expected_solution": "f(x) = 2x - 3 oldugundan f(f(4)) icin once f(4)=5 bulunur. Sonra f(5)=7 olur.",
        "image_path": "data/comparison_questions/cmp_m01_function.png",
        "prompt": [
            "f(x) = 2x - 3 fonksiyonu veriliyor.",
            "Buna gore f(f(4)) kactir?",
            "A) 3    B) 5    C) 7    D) 9    E) 11",
        ],
    },
    {
        "question_id": "cmp_m02",
        "subject": "matematik",
        "topic": "parabol grafiği",
        "difficulty": "orta",
        "correct_answer": "D",
        "final_answer": "-1",
        "expected_solution": "Grafikte parabolun tepe noktasi (2, -1) olarak verilmistir. Minimum deger tepe noktasinin y degeridir.",
        "image_path": "data/comparison_questions/cmp_m02_parabola_graph.png",
    },
    {
        "question_id": "cmp_m03",
        "subject": "matematik",
        "topic": "olasılık",
        "difficulty": "orta",
        "correct_answer": "B",
        "final_answer": "1/3",
        "expected_solution": "Kutuda 3 kirmizi, 2 mavi, 1 yesil top vardir. Toplam 6 top icinde mavi top sayisi 2 oldugundan olasilik 2/6 = 1/3 olur.",
        "image_path": "data/comparison_questions/cmp_m03_probability.png",
        "prompt": [
            "Bir kutuda 3 kirmizi, 2 mavi ve 1 yesil top vardir.",
            "Rastgele secilen bir topun mavi olma olasiligi nedir?",
            "A) 1/6    B) 1/3    C) 1/2    D) 2/3    E) 5/6",
        ],
    },
    {
        "question_id": "cmp_m04",
        "subject": "matematik",
        "topic": "geometri - alan",
        "difficulty": "orta",
        "correct_answer": "C",
        "final_answer": "6",
        "expected_solution": "Dik ucgende alan (dik kenarlar carpimi)/2'dir. 4*x/2 = 12 den x = 6 bulunur.",
        "image_path": "data/comparison_questions/cmp_m04_triangle_area.png",
    },
    {
        "question_id": "cmp_m05",
        "subject": "matematik",
        "topic": "trigonometri",
        "difficulty": "orta",
        "correct_answer": "C",
        "final_answer": "5",
        "expected_solution": "sin A = karsi/hipotenus = 3/5 ise 3-4-5 dik ucgeninden cos A = 4/5 olur. 5*cos A + 1 = 4 + 1 = 5.",
        "image_path": "data/comparison_questions/cmp_m05_trigonometry.png",
        "prompt": [
            "Dar aci A icin sin A = 3/5 veriliyor.",
            "Buna gore 5*cos A + 1 kactir?",
            "A) 3    B) 4    C) 5    D) 6    E) 7",
        ],
    },
    {
        "question_id": "cmp_m06",
        "subject": "matematik",
        "topic": "aritmetik dizi",
        "difficulty": "orta",
        "correct_answer": "D",
        "final_answer": "25",
        "expected_solution": "Aritmetik dizide ortak fark 3'tur. a7 = a1 + 6d = 7 + 18 = 25 olur.",
        "image_path": "data/comparison_questions/cmp_m06_sequence.png",
        "prompt": [
            "Bir aritmetik dizide a1 = 7 ve a4 = 16'dır.",
            "Buna gore a7 kactir?",
            "A) 19    B) 21    C) 23    D) 25    E) 27",
        ],
    },
    {
        "question_id": "cmp_m07",
        "subject": "matematik",
        "topic": "türev",
        "difficulty": "orta",
        "correct_answer": "B",
        "final_answer": "8",
        "expected_solution": "f'(x) = 3x^2 - 4x oldugundan f'(2) = 12 - 8 = 4 degil; soru f(x)=x^3-2x^2+4x icin f'(x)=3x^2-4x+4, f'(2)=12-8+4=8 olur.",
        "image_path": "data/comparison_questions/cmp_m07_derivative.png",
        "prompt": [
            "f(x) = x^3 - 2x^2 + 4x fonksiyonu veriliyor.",
            "Buna gore f'(2) kactir?",
            "A) 6    B) 8    C) 10    D) 12    E) 14",
        ],
    },
    {
        "question_id": "cmp_m08",
        "subject": "matematik",
        "topic": "belirli integral",
        "difficulty": "orta",
        "correct_answer": "E",
        "final_answer": "10",
        "expected_solution": "Integral 0'dan 2'ye (3x+2) dx = [(3/2)x^2 + 2x]_0^2 = 6 + 4 = 10 olur.",
        "image_path": "data/comparison_questions/cmp_m08_integral_area.png",
    },
    {
        "question_id": "cmp_m09",
        "subject": "matematik",
        "topic": "logaritma",
        "difficulty": "orta",
        "correct_answer": "C",
        "final_answer": "5",
        "expected_solution": "log2(x-1)=3 ise x-1=2^3=8, buradan x=9 olur. Koku sorulan ifade x-4 oldugundan 5'tir.",
        "image_path": "data/comparison_questions/cmp_m09_logarithm.png",
        "prompt": [
            "log2(x - 1) = 3 olduguna gore",
            "x - 4 kactir?",
            "A) 3    B) 4    C) 5    D) 6    E) 8",
        ],
    },
    {
        "question_id": "cmp_m10",
        "subject": "matematik",
        "topic": "tablo yorumlama",
        "difficulty": "orta",
        "correct_answer": "D",
        "final_answer": "42",
        "expected_solution": "2 defter 36 TL ve 3 kalem 6 TL tuttugundan toplam 42 TL olur.",
        "image_path": "data/comparison_questions/cmp_m10_table.png",
    },
    {
        "question_id": "cmp_p01",
        "subject": "fizik",
        "topic": "hız-zaman grafiği",
        "difficulty": "orta",
        "correct_answer": "C",
        "final_answer": "40 m",
        "expected_solution": "Hiz-zaman grafiginin altindaki alan yer degistirmeyi verir. 0-4 s arasi hiz 10 m/s oldugundan alan 4*10=40 m'dir.",
        "image_path": "data/comparison_questions/cmp_p01_velocity_time.png",
    },
    {
        "question_id": "cmp_p02",
        "subject": "fizik",
        "topic": "dinamik",
        "difficulty": "orta",
        "correct_answer": "B",
        "final_answer": "3 m/s^2",
        "expected_solution": "Net kuvvet 20 - 8 = 12 N'dur. F = m*a bagintisindan a = 12/4 = 3 m/s^2 bulunur.",
        "image_path": "data/comparison_questions/cmp_p02_dynamics.png",
    },
    {
        "question_id": "cmp_p03",
        "subject": "fizik",
        "topic": "iş ve enerji",
        "difficulty": "orta",
        "correct_answer": "D",
        "final_answer": "80 J",
        "expected_solution": "Yatay duzlemde kuvvet hareket dogrultusunda oldugundan W = F*x = 16*5 = 80 J olur.",
        "image_path": "data/comparison_questions/cmp_p03_work.png",
        "prompt": [
            "Yatay duzlemde duran bir cisme hareket dogrultusunda",
            "16 N kuvvet uygulanip cisim 5 m yer degistiriyor.",
            "Yapilan is kactir?",
            "A) 20 J    B) 40 J    C) 60 J    D) 80 J    E) 100 J",
        ],
    },
    {
        "question_id": "cmp_p04",
        "subject": "fizik",
        "topic": "elektrik devresi",
        "difficulty": "orta",
        "correct_answer": "B",
        "final_answer": "2 A",
        "expected_solution": "Seri bagli 4 ohm ve 2 ohm direncin esdegeri 6 ohm'dur. I = V/R = 12/6 = 2 A.",
        "image_path": "data/comparison_questions/cmp_p04_circuit.png",
    },
    {
        "question_id": "cmp_p05",
        "subject": "fizik",
        "topic": "optik - aynalar",
        "difficulty": "orta",
        "correct_answer": "D",
        "final_answer": "20 cm",
        "expected_solution": "Duzlem aynada goruntunun aynaya uzakligi cismin aynaya uzakligina esittir. Cisim aynadan 20 cm uzaktaysa goruntu de 20 cm uzaktadir.",
        "image_path": "data/comparison_questions/cmp_p05_mirror.png",
    },
    {
        "question_id": "cmp_p06",
        "subject": "fizik",
        "topic": "dalga",
        "difficulty": "orta",
        "correct_answer": "C",
        "final_answer": "12 m/s",
        "expected_solution": "Dalga hizi v = lambda*f bagintisiyle bulunur. 3 m * 4 Hz = 12 m/s.",
        "image_path": "data/comparison_questions/cmp_p06_wave.png",
        "prompt": [
            "Bir dalganin dalga boyu 3 m, frekansi 4 Hz'dir.",
            "Dalganin yayilma hizi kactir?",
            "A) 7 m/s    B) 9 m/s    C) 12 m/s    D) 15 m/s    E) 18 m/s",
        ],
    },
    {
        "question_id": "cmp_p07",
        "subject": "fizik",
        "topic": "kaldırma kuvveti",
        "difficulty": "orta",
        "correct_answer": "A",
        "final_answer": "30 N",
        "expected_solution": "Cisim sivi icinde dengede yuzdugune gore kaldirma kuvveti cismin agirligina esittir, yani 30 N.",
        "image_path": "data/comparison_questions/cmp_p07_buoyancy.png",
    },
    {
        "question_id": "cmp_p08",
        "subject": "fizik",
        "topic": "ısı ve sıcaklık",
        "difficulty": "orta",
        "correct_answer": "E",
        "final_answer": "4200 J",
        "expected_solution": "Q = m*c*DeltaT. m=0,1 kg, c=4200 J/kg C ve DeltaT=10 C oldugundan Q=0,1*4200*10=4200 J.",
        "image_path": "data/comparison_questions/cmp_p08_heat.png",
        "prompt": [
            "100 g suyun sicakligi 20 C'den 30 C'ye cikariliyor.",
            "c_su = 4200 J/(kg C) olduguna gore alinan isi kactir?",
            "A) 420 J    B) 840 J    C) 2100 J    D) 3000 J    E) 4200 J",
        ],
    },
    {
        "question_id": "cmp_p09",
        "subject": "fizik",
        "topic": "momentum",
        "difficulty": "orta",
        "correct_answer": "B",
        "final_answer": "4 m/s",
        "expected_solution": "Tam esnek olmayan carpisma sonrasi ortak hiz v = (m1*v1 + m2*v2)/(m1+m2) = (2*6 + 1*0)/3 = 4 m/s.",
        "image_path": "data/comparison_questions/cmp_p09_momentum.png",
    },
    {
        "question_id": "cmp_p10",
        "subject": "fizik",
        "topic": "tork",
        "difficulty": "orta",
        "correct_answer": "C",
        "final_answer": "24 N.m",
        "expected_solution": "Kuvvet cubuga dik uygulandigi icin tork = F*d = 12*2 = 24 N.m olur.",
        "image_path": "data/comparison_questions/cmp_p10_torque.png",
    },
]


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


def _new_canvas(width: int = 1400, height: int = 900) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def _line_height(font: ImageFont.ImageFont, spacing: int = 10) -> int:
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        return ascent + descent + spacing
    return 36 + spacing


def _draw_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    width: int = 58,
    spacing: int = 10,
) -> int:
    current_y = y
    for line in lines:
        wrapped = textwrap.wrap(line, width=width) or [""]
        for wrapped_line in wrapped:
            draw.text((x, current_y), wrapped_line, fill="black", font=font)
            current_y += _line_height(font, spacing)
    return current_y


def _draw_standard_text_question(output_path: Path, lines: list[str]) -> None:
    image, draw = _new_canvas()
    font = _load_font(42)
    _draw_wrapped_lines(draw, lines, 70, 80, font, width=62, spacing=12)
    image.save(output_path)


def _axis(draw: ImageDraw.ImageDraw, origin: tuple[int, int], x_len: int, y_len: int, font: ImageFont.ImageFont) -> None:
    ox, oy = origin
    draw.line((ox, oy, ox + x_len, oy), fill="black", width=3)
    draw.line((ox, oy, ox, oy - y_len), fill="black", width=3)
    draw.text((ox + x_len + 12, oy - 18), "x", fill="black", font=font)
    draw.text((ox - 12, oy - y_len - 42), "y", fill="black", font=font)


def _point(origin: tuple[int, int], scale: int, x_value: float, y_value: float) -> tuple[int, int]:
    ox, oy = origin
    return int(ox + x_value * scale), int(oy - y_value * scale)


def draw_m02(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(34)
    origin = (360, 500)
    scale = 80
    _axis(draw, origin, 550, 360, font)
    for x in range(-2, 6):
        px, py = _point(origin, scale, x, 0)
        draw.line((px, py - 7, px, py + 7), fill="black", width=2)
        draw.text((px - 8, py + 14), str(x), fill="black", font=_load_font(22))
    for y in range(-2, 4):
        px, py = _point(origin, scale, 0, y)
        draw.line((px - 7, py, px + 7, py), fill="black", width=2)
        if y != 0:
            draw.text((px - 38, py - 15), str(y), fill="black", font=_load_font(22))
    points = [_point(origin, scale, -0.5 + i * 0.05, (-0.5 + i * 0.05 - 2) ** 2 - 1) for i in range(101)]
    draw.line(points, fill="#1f66cc", width=5)
    vx, vy = _point(origin, scale, 2, -1)
    draw.ellipse((vx - 8, vy - 8, vx + 8, vy + 8), fill="#c00000")
    draw.text((vx + 18, vy - 8), "T(2, -1)", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["Grafikte verilen parabolun minimum degeri kactir?", "A) -4    B) -3    C) -2    D) -1    E) 0"],
        70,
        670,
        _load_font(40),
        width=60,
    )
    image.save(output_path)


def draw_m04(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(40)
    tri = [(240, 590), (240, 250), (660, 590)]
    draw.polygon(tri, outline="black", fill="#f4fbff")
    draw.line(tri + [tri[0]], fill="black", width=5)
    draw.line((240, 545, 285, 545, 285, 590), fill="black", width=3)
    draw.text((410, 610), "4", fill="black", font=font)
    draw.text((165, 400), "x", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["Dik ucgenin alani 12 birim kare olduguna gore x kactir?", "A) 3    B) 4    C) 6    D) 8    E) 10"],
        70,
        90,
        font,
        width=62,
    )
    image.save(output_path)


def draw_m08(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(38)
    origin = (240, 600)
    scale = 90
    _axis(draw, origin, 540, 420, font)
    pts = [_point(origin, scale, x / 20, 3 * (x / 20) + 2) for x in range(0, 41)]
    fill_poly = [_point(origin, scale, 0, 0)] + pts + [_point(origin, scale, 2, 0)]
    draw.polygon(fill_poly, fill="#e8f3ff", outline="#1f66cc")
    draw.line(pts, fill="#1f66cc", width=5)
    draw.text((origin[0] + 2 * scale - 8, origin[1] + 16), "2", fill="black", font=_load_font(24))
    draw.text((origin[0] - 45, origin[1] - 2 * scale - 12), "2", fill="black", font=_load_font(24))
    draw.text((origin[0] + 2 * scale + 12, origin[1] - 8 * scale - 20), "y=3x+2", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["Sekilde boyali bolgenin alani kactir?", "A) 4    B) 6    C) 8    D) 9    E) 10"],
        70,
        90,
        font,
        width=62,
    )
    image.save(output_path)


def draw_m10(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(36)
    _draw_wrapped_lines(draw, ["Kirtasiye fiyat tablosu:"], 70, 70, _load_font(42))
    x0, y0 = 180, 180
    widths = [240, 220]
    rows = [("Urun", "Fiyat"), ("Defter", "18 TL"), ("Kalem", "2 TL"), ("Silgi", "5 TL")]
    for r, row in enumerate(rows):
        y = y0 + r * 80
        for c, text in enumerate(row):
            x = x0 + sum(widths[:c])
            fill = "#eef5ff" if r == 0 else "white"
            draw.rectangle((x, y, x + widths[c], y + 80), fill=fill, outline="black", width=3)
            draw.text((x + 26, y + 20), text, fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["2 defter ve 3 kalem alan bir ogrenci toplam kac TL oder?", "A) 36    B) 38    C) 40    D) 42    E) 45"],
        70,
        560,
        _load_font(40),
        width=62,
    )
    image.save(output_path)


def draw_p01(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(34)
    origin = (230, 590)
    scale_x, scale_y = 120, 35
    draw.line((origin[0], origin[1], origin[0] + 620, origin[1]), fill="black", width=3)
    draw.line((origin[0], origin[1], origin[0], origin[1] - 440), fill="black", width=3)
    draw.text((origin[0] + 640, origin[1] - 20), "t(s)", fill="black", font=font)
    draw.text((origin[0] - 35, origin[1] - 485), "v(m/s)", fill="black", font=font)
    y10 = origin[1] - 10 * scale_y
    draw.line((origin[0], y10, origin[0] + 4 * scale_x, y10), fill="#1f66cc", width=5)
    draw.line((origin[0] + 4 * scale_x, y10, origin[0] + 4 * scale_x, origin[1]), fill="#1f66cc", width=3)
    draw.text((origin[0] - 45, y10 - 15), "10", fill="black", font=_load_font(26))
    draw.text((origin[0] + 4 * scale_x - 8, origin[1] + 16), "4", fill="black", font=_load_font(26))
    _draw_wrapped_lines(
        draw,
        ["Hiz-zaman grafigine gore 0-4 s araliginda cismin yer degistirmesi kactir?", "A) 10 m    B) 20 m    C) 40 m    D) 50 m    E) 80 m"],
        70,
        70,
        _load_font(40),
        width=62,
    )
    image.save(output_path)


def draw_p02(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(38)
    draw.rectangle((470, 380, 760, 550), fill="#f6f6f6", outline="black", width=5)
    draw.text((555, 435), "m=4 kg", fill="black", font=font)
    draw.line((760, 465, 980, 465), fill="#1f66cc", width=6)
    draw.polygon([(980, 465), (940, 445), (940, 485)], fill="#1f66cc")
    draw.text((840, 405), "20 N", fill="black", font=font)
    draw.line((470, 465, 300, 465), fill="#c00000", width=6)
    draw.polygon([(300, 465), (340, 445), (340, 485)], fill="#c00000")
    draw.text((325, 405), "8 N", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["Surtunmesiz yatay duzlemdeki cisme sekildeki kuvvetler uygulanmaktadir.", "Cismin ivmesi kactir?", "A) 2 m/s^2    B) 3 m/s^2    C) 4 m/s^2    D) 5 m/s^2    E) 7 m/s^2"],
        70,
        70,
        font,
        width=62,
    )
    image.save(output_path)


def draw_p04(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(36)
    left, top, right, bottom = 220, 250, 920, 600
    draw.line((left, top, left, bottom, right, bottom, right, top, left, top), fill="black", width=4)
    draw.rectangle((430, top - 40, 560, top + 40), outline="black", width=4, fill="#fff8e8")
    draw.text((445, top - 25), "4 ohm", fill="black", font=_load_font(28))
    draw.rectangle((680, top - 40, 810, top + 40), outline="black", width=4, fill="#fff8e8")
    draw.text((695, top - 25), "2 ohm", fill="black", font=_load_font(28))
    draw.line((left - 25, 410, left + 25, 410), fill="black", width=4)
    draw.line((left - 12, 445, left + 12, 445), fill="black", width=4)
    draw.text((105, 392), "12 V", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["Seri bagli direnclerden olusan devrede akim siddeti kactir?", "A) 1 A    B) 2 A    C) 3 A    D) 4 A    E) 6 A"],
        70,
        70,
        _load_font(40),
        width=62,
    )
    image.save(output_path)


def draw_p05(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(38)
    draw.line((700, 210, 700, 660), fill="black", width=5)
    for y in range(220, 660, 35):
        draw.line((700, y, 735, y - 25), fill="gray", width=2)
    draw.ellipse((430, 380, 510, 460), fill="#ffe7b8", outline="black", width=3)
    draw.line((470, 460, 470, 580), fill="black", width=4)
    draw.line((470, 500, 420, 540), fill="black", width=4)
    draw.line((470, 500, 520, 540), fill="black", width=4)
    draw.text((535, 410), "20 cm", fill="black", font=font)
    draw.line((510, 430, 695, 430), fill="#1f66cc", width=4)
    _draw_wrapped_lines(
        draw,
        ["Duzlem aynanin onundeki cisim aynadan 20 cm uzaktadir.", "Goruntunun aynaya uzakligi kac cm'dir?", "A) 5    B) 10    C) 15    D) 20    E) 40"],
        70,
        70,
        font,
        width=62,
    )
    image.save(output_path)


def draw_p07(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(38)
    draw.rectangle((260, 390, 900, 720), fill="#dff3ff", outline="black", width=4)
    draw.rectangle((500, 300, 690, 470), fill="#ffd9a8", outline="black", width=4)
    draw.line((500, 390, 690, 390), fill="#1f66cc", width=4)
    draw.line((595, 300, 595, 210), fill="#c00000", width=5)
    draw.polygon([(595, 210), (575, 250), (615, 250)], fill="#c00000")
    draw.text((615, 225), "F_k", fill="black", font=font)
    draw.line((595, 470, 595, 595), fill="#333333", width=5)
    draw.polygon([(595, 595), (575, 555), (615, 555)], fill="#333333")
    draw.text((615, 530), "G=30 N", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["Sivi icinde dengede yuzen cismin agirligi 30 N'dur.", "Cisme etki eden kaldirma kuvveti kactir?", "A) 30 N    B) 20 N    C) 15 N    D) 10 N    E) 5 N"],
        70,
        70,
        font,
        width=62,
    )
    image.save(output_path)


def draw_p09(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(38)
    draw.rectangle((280, 420, 430, 540), fill="#e8f3ff", outline="black", width=4)
    draw.text((305, 460), "2 kg", fill="black", font=font)
    draw.line((430, 480, 610, 480), fill="#1f66cc", width=6)
    draw.polygon([(610, 480), (570, 460), (570, 500)], fill="#1f66cc")
    draw.text((450, 420), "6 m/s", fill="black", font=font)
    draw.rectangle((760, 420, 880, 540), fill="#fff2cc", outline="black", width=4)
    draw.text((785, 460), "1 kg", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["2 kg kutleli cisim 6 m/s hizla durmakta olan 1 kg kutleli cisme carpip birlikte hareket ediyor.", "Ortak hiz kactir?", "A) 3 m/s    B) 4 m/s    C) 5 m/s    D) 6 m/s    E) 8 m/s"],
        70,
        70,
        font,
        width=62,
    )
    image.save(output_path)


def draw_p10(output_path: Path) -> None:
    image, draw = _new_canvas()
    font = _load_font(38)
    draw.ellipse((215, 485, 265, 535), fill="black")
    draw.line((240, 510, 860, 510), fill="#8b5a2b", width=18)
    draw.line((720, 510, 720, 300), fill="#c00000", width=6)
    draw.polygon([(720, 300), (700, 340), (740, 340)], fill="#c00000")
    draw.text((740, 325), "12 N", fill="black", font=font)
    draw.line((240, 560, 720, 560), fill="#1f66cc", width=4)
    draw.text((440, 585), "2 m", fill="black", font=font)
    _draw_wrapped_lines(
        draw,
        ["Noktasi etrafinda donebilen cubuga sekildeki gibi dik 12 N kuvvet uygulanir.", "Olusan tork kac N.m'dir?", "A) 12    B) 18    C) 24    D) 30    E) 36"],
        70,
        70,
        font,
        width=62,
    )
    image.save(output_path)


DRAWERS: dict[str, Callable[[Path], None]] = {
    "cmp_m02": draw_m02,
    "cmp_m04": draw_m04,
    "cmp_m08": draw_m08,
    "cmp_m10": draw_m10,
    "cmp_p01": draw_p01,
    "cmp_p02": draw_p02,
    "cmp_p04": draw_p04,
    "cmp_p05": draw_p05,
    "cmp_p07": draw_p07,
    "cmp_p09": draw_p09,
    "cmp_p10": draw_p10,
}


def _metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    question_id = record["question_id"]
    return {
        "id": question_id,
        "question_id": question_id,
        "subject": record["subject"],
        "topic": record["topic"],
        "question_type": f"{record['subject']}_{record['topic']}",
        "difficulty": record["difficulty"],
        "correct_answer": record["correct_answer"],
        "expected_solution": record["expected_solution"],
        "expected_reasoning": record["expected_solution"],
        "final_answer": record["final_answer"],
        "image_path": record["image_path"],
    }


def create_all_comparison_questions(output_dir: Path = COMPARISON_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    for record in QUESTION_RECORDS:
        output_path = Path(record["image_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drawer = DRAWERS.get(record["question_id"])
        if drawer is None:
            _draw_standard_text_question(output_path, record["prompt"])
        else:
            drawer(output_path)
        created_paths.append(output_path)

    metadata = [_metadata_record(record) for record in QUESTION_RECORDS]
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return created_paths


if __name__ == "__main__":
    paths = create_all_comparison_questions()
    for path in paths:
        print(f"Created comparison question image at: {path}")
    print(f"Wrote metadata to: {METADATA_PATH}")
