from services.adaptive_router import decide_pipeline


def test_turkish_paragraph_question_routes_to_ocr() -> None:
    result = decide_pipeline(
        "A\u015fa\u011f\u0131daki paragrafta yazar\u0131n ana d\u00fc\u015f\u00fcncesi hangisidir?",
        ocr_confidence=0.91,
    )

    assert result["recommended_mode"] == "ocr"
    assert result["detected_subject"] == "turkish"
    assert result["detected_question_type"] == "text_only"


def test_social_history_question_routes_to_ocr() -> None:
    result = decide_pipeline(
        "Osmanl\u0131 Devleti'nin I. D\u00fcnya Sava\u015f\u0131'na girmesinin sonu\u00e7lar\u0131ndan biri hangisidir?",
        ocr_confidence=0.88,
    )

    assert result["recommended_mode"] == "ocr"
    assert result["detected_subject"] == "social"
    assert result["detected_question_type"] == "social_text"


def test_simple_math_text_detects_math_subject() -> None:
    result = decide_pipeline("Bir say\u0131n\u0131n 3 fazlas\u0131 12 ise bu say\u0131 ka\u00e7t\u0131r?", ocr_confidence=0.9)

    assert result["recommended_mode"] in {"ocr", "both"}
    assert result["detected_subject"] == "math"
    assert result["detected_question_type"] == "math_text"


def test_plain_math_word_problem_routes_to_ocr() -> None:
    result = decide_pipeline(
        "Mina has 8 pencils and buys 7 more. How many pencils does she have now?\n"
        "A) 12\nB) 13\nC) 14\nD) 15\nE) 16",
        ocr_confidence=0.9,
    )

    assert result["recommended_mode"] == "ocr"
    assert result["detected_question_type"] == "math_text"


def test_geometry_question_routes_to_vision() -> None:
    result = decide_pipeline("\u015eekildeki \u00fc\u00e7gende A a\u00e7\u0131s\u0131 ka\u00e7 derecedir?", ocr_confidence=0.86)

    assert result["recommended_mode"] == "vision"
    assert result["detected_question_type"] == "geometry"


def test_chart_table_question_routes_to_vision() -> None:
    result = decide_pipeline("Tablodaki verilere g\u00f6re en y\u00fcksek de\u011fer hangisidir?", ocr_confidence=0.83)

    assert result["recommended_mode"] == "vision"
    assert result["detected_question_type"] == "chart_table"


def test_team_points_bar_chart_does_not_route_to_ocr() -> None:
    result = decide_pipeline(
        "Team Points\n8\n6\n5\nRed Blue Green\n\n"
        "Which team has the highest number of points?\n"
        "A) Red\nB) Blue\nC) Green\nD) Yellow\nE) All equal",
        ocr_confidence=0.9,
    )

    assert result["recommended_mode"] in {"vision", "both"}
    assert result["recommended_mode"] != "ocr"
    assert result["detected_question_type"] == "chart_table"


def test_coordinate_plane_point_p_does_not_route_to_ocr() -> None:
    result = decide_pipeline(
        "Coordinate Plane\n"
        "Point P is shown on the grid. What are its coordinates?\n"
        "A) (2, 3)\nB) (3, 2)\nC) (-3, 2)",
        ocr_confidence=0.9,
    )

    assert result["recommended_mode"] in {"vision", "both"}
    assert result["recommended_mode"] != "ocr"
    assert result["detected_question_type"] == "geometry"


def test_visual_question_with_uncertain_ocr_routes_to_both() -> None:
    result = decide_pipeline(
        "Team Points\n8\n6\n5\nRed Blue Green\nWhich team has the highest number of points?",
        ocr_confidence=0.5,
    )

    assert result["recommended_mode"] == "both"


def test_low_ocr_confidence_routes_to_both() -> None:
    result = decide_pipeline("Bu metin okunmu\u015f gibi g\u00f6r\u00fcnse de g\u00fcven d\u00fc\u015f\u00fck.", ocr_confidence=0.12)

    assert result["recommended_mode"] == "both"
    assert result["detected_subject"] == "unknown"


def test_empty_unknown_text_routes_to_both() -> None:
    result = decide_pipeline("", ocr_confidence=None)

    assert result["recommended_mode"] == "both"
    assert result["detected_subject"] == "unknown"
    assert result["detected_question_type"] == "unknown"


def test_turkish_inference_options_do_not_trigger_geometry_route() -> None:
    result = decide_pipeline(
        "Elif yagmuru gorunce semsiyesini cantasina koydu. Bu durumdan hangi sonuc cikarilir?\n"
        "A) Hava sicaktir\n"
        "B) Elif hazirliklidir\n"
        "C) Canta bostur\n"
        "D) Yol kisadir\n"
        "E) Elif acikmistir",
        ocr_confidence=0.9,
    )

    assert result["recommended_mode"] == "ocr"
    assert result["detected_subject"] == "turkish"


def test_social_question_option_geometry_does_not_trigger_visual_route() -> None:
    result = decide_pipeline(
        "A town council keeps written records of decisions so later generations can learn what happened. "
        "This most directly helps which field?\n"
        "A) History\n"
        "B) Geometry\n"
        "C) Music\n"
        "D) Astronomy\n"
        "E) Chemistry",
        ocr_confidence=0.9,
    )

    assert result["recommended_mode"] == "ocr"
    assert result["detected_subject"] == "social"
