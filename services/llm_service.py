from typing import Any, Dict


def solve_text_question(question_text: str) -> Dict[str, Any]:
    """Use a language model to solve a question expressed in text."""
    return {
        "question_text": question_text,
        "solution": "",
        "metadata": {},
    }


def solve_image_question_direct(image_path: str) -> Dict[str, Any]:
    """Use a vision-capable LLM to solve a question from an image input."""
    return {
        "image_path": image_path,
        "solution": "",
        "metadata": {},
    }
