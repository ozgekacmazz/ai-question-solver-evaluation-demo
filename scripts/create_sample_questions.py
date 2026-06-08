from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_default_font() -> ImageFont.ImageFont:
    candidate = Path("C:/Windows/Fonts/arial.ttf")
    if candidate.exists():
        try:
            return ImageFont.truetype(str(candidate), 48)
        except Exception:
            pass
    return ImageFont.load_default()


def create_sample_question_image(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 500
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = _load_default_font()

    text_lines = [
        "What is 2 + 2?",
        "A) 3",
        "B) 4",
        "C) 5",
    ]

    line_spacing = 16
    current_y = 60
    left_margin = 60

    for line in text_lines:
        draw.text((left_margin, current_y), line, fill="black", font=font)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent + line_spacing
        current_y += line_height

    image.save(output_path)


if __name__ == "__main__":
    destination = Path("data") / "sample_questions" / "q01_text.png"
    create_sample_question_image(destination)
    print(f"Created sample question image at: {destination}")
