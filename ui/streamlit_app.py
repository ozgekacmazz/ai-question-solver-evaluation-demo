import streamlit as st


def main() -> None:
    """Streamlit UI entry point for the evaluation demo."""
    st.title("AI Question Solver Evaluation Demo")
    st.write(
        "Compare OCR + LLM and direct Vision LLM approaches for solving question images."
    )

    st.sidebar.header("Settings")
    st.sidebar.selectbox("Pipeline", ["ocr_llm", "vision_llm"])

    st.write("This demo is a skeleton app for future UI development.")


if __name__ == "__main__":
    main()
