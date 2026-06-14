from __future__ import annotations

import re
from typing import Any


LOW_OCR_CONFIDENCE_THRESHOLD = 0.35

TURKISH_CHAR_MAP = str.maketrans(
    {
        "\u00e7": "c",
        "\u011f": "g",
        "\u0131": "i",
        "\u00f6": "o",
        "\u015f": "s",
        "\u00fc": "u",
        "\u00e2": "a",
        "\u00ee": "i",
        "\u00fb": "u",
        "\u0307": "",
    }
)

GEOMETRY_KEYWORDS = {
    "triangle",
    "angle",
    "circle",
    "coordinate plane",
    "coordinate",
    "rectangle",
    "square",
    "geometry",
    "shape",
    "grid",
    "point p",
    "shown on the grid",
    "diagram",
    "figure",
    "sekil",
    "ucgen",
    "aci",
    "cember",
    "daire",
    "dikdortgen",
    "kare",
    "geometri",
}

CHART_TABLE_KEYWORDS = {
    "graph",
    "chart",
    "table",
    "bar chart",
    "bar",
    "line graph",
    "axis",
    "team points",
    "grafik",
    "tablo",
    "sutun grafigi",
}

ADVANCED_MATH_KEYWORDS = {
    "derivative",
    "integral",
    "limit",
    "parabola",
    "function",
    "turev",
    "parabol",
    "fonksiyon",
    "x^2",
    "x\u00b2",
}

MATH_KEYWORDS = {
    "math",
    "matematik",
    "equation",
    "denklem",
    "how many",
    "sum",
    "toplam",
    "fraction",
    "kesir",
    "number",
    "sayi",
    "calculate",
    "more",
    "hesaplayiniz",
    "islem",
}

SCIENCE_KEYWORDS = {
    "science",
    "fen",
    "physics",
    "fizik",
    "chemistry",
    "kimya",
    "biology",
    "biyoloji",
    "cell",
    "hucre",
    "force",
    "kuvvet",
    "energy",
    "enerji",
}

SOCIAL_KEYWORDS = {
    "history",
    "geography",
    "social",
    "written records",
    "town council",
    "later generations",
    "field",
    "climate",
    "region",
    "rain",
    "sparse plants",
    "votes",
    "participation",
    "tarih",
    "cografya",
    "sosyal",
    "osmanli",
    "selcuklu",
    "cumhuriyet",
    "savas",
    "antlasma",
    "iklim",
    "nufus",
    "harita",
    "bolge",
    "ekonomi",
}

TURKISH_TEXT_KEYWORDS = {
    "turkce",
    "paragraf",
    "metin",
    "cumle",
    "sozcuk",
    "anlam",
    "ana fikir",
    "yardimci fikir",
    "yazar",
    "dusunce",
    "cikarim",
    "sonuc cikarilir",
    "hangi sonuc",
    "durumdan",
    "yagmuru",
    "semsiyesini",
    "hazirliklidir",
    "asagidakilerden",
    "hangisi",
}


def _normalize_text(text: str) -> tuple[str, str]:
    lowered = (text or "").strip().lower()
    ascii_text = lowered.translate(TURKISH_CHAR_MAP)
    return lowered, ascii_text


def _contains_any(text: str, keywords: set[str]) -> bool:
    for keyword in keywords:
        if " " in keyword or len(keyword) > 3:
            if keyword in text:
                return True
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False


def _routing_text(text: str) -> str:
    """Use the stem for routing so answer options do not masquerade as question signals."""
    return re.split(r"(?im)^\s*A\s*[\).]", text or "", maxsplit=1)[0]


def _safe_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result(
    detected_subject: str,
    detected_question_type: str,
    recommended_mode: str,
    reason: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "detected_subject": detected_subject,
        "detected_question_type": detected_question_type,
        "recommended_mode": recommended_mode,
        "reason": reason,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
    }


def _visual_mode(confidence_value: float | None) -> str:
    if confidence_value is not None and confidence_value < 0.65:
        return "both"
    return "vision"


def decide_pipeline(ocr_text: str, ocr_confidence: float | None = None) -> dict[str, Any]:
    """Choose the safest solve pipeline from OCR text and optional OCR confidence."""
    _, text = _normalize_text(_routing_text(ocr_text))
    confidence_value = _safe_confidence(ocr_confidence)

    if confidence_value is not None and confidence_value < LOW_OCR_CONFIDENCE_THRESHOLD:
        return _result(
            "unknown",
            "unknown",
            "both",
            "OCR confidence is very low, so both OCR and vision should be compared.",
            0.9,
        )

    if not text:
        return _result(
            "unknown",
            "unknown",
            "both",
            "OCR text is empty or unavailable.",
            0.85,
        )

    has_geometry = _contains_any(text, GEOMETRY_KEYWORDS)
    has_chart_table = _contains_any(text, CHART_TABLE_KEYWORDS)
    has_visual = has_geometry or has_chart_table
    has_advanced_math = _contains_any(text, ADVANCED_MATH_KEYWORDS)
    has_math = (
        has_advanced_math
        or _contains_any(text, MATH_KEYWORDS)
        or any(symbol in text for symbol in ("+", "-", "=", " x "))
    )
    has_science = _contains_any(text, SCIENCE_KEYWORDS)
    has_social = _contains_any(text, SOCIAL_KEYWORDS)
    has_turkish_text = _contains_any(text, TURKISH_TEXT_KEYWORDS)

    if has_geometry:
        recommended_mode = _visual_mode(confidence_value)
        return _result(
            "math",
            "geometry",
            recommended_mode,
            "Geometry or figure-related keywords suggest visual context is important.",
            0.92,
        )

    if has_chart_table:
        subject = "math" if has_math else "unknown"
        recommended_mode = _visual_mode(confidence_value)
        return _result(
            subject,
            "chart_table",
            recommended_mode,
            "Graph, chart, or table keywords suggest the image layout matters.",
            0.92,
        )

    if has_advanced_math:
        return _result(
            "math",
            "visual_math" if has_visual else "math_text",
            "vision" if has_visual else "both",
            "Advanced math often needs visual checks unless the OCR text is clearly sufficient.",
            0.78,
        )

    if has_social:
        return _result(
            "social",
            "social_text",
            "ocr",
            "Social studies or history/geography keywords point to a text-heavy question.",
            0.88,
        )

    if has_science:
        return _result(
            "science",
            "science_text",
            "ocr",
            "Science keywords appear in a text-heavy question.",
            0.82,
        )

    if has_turkish_text:
        return _result(
            "turkish",
            "text_only",
            "ocr",
            "Turkish language or paragraph reasoning keywords point to OCR text solving.",
            0.88,
        )

    if has_math:
        return _result(
            "math",
            "math_text",
            "ocr",
            "The question appears to be a plain math text problem.",
            0.74,
        )

    return _result(
        "unknown",
        "unknown",
        "both",
        "No strong text or visual routing signal was detected.",
        0.65,
    )
