from services.llm_service import solve_text_question, solve_image_question_direct


def test_solve_text_question_returns_solution_key() -> None:
    result = solve_text_question("What is 2 + 2?")
    assert "solution" in result


def test_solve_image_question_direct_returns_solution_key() -> None:
    result = solve_image_question_direct("dummy_path.png")
    assert "solution" in result
