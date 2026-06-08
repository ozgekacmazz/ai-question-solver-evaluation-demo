import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from services.solver_pipeline import solve_question_image


def main() -> None:
    """Streamlit UI entry point for the evaluation demo."""
    st.title("AI Question Solver Evaluation Demo")
    st.info(
        "This demo compares AI-based approaches for solving question images. "
        "Currently, the active pipeline is OCR + Mock LLM."
    )

    st.sidebar.header("Project status")
    st.sidebar.write("OCR preprocessing pipeline works.")
    st.sidebar.write("Current mode: OCR + Mock LLM")
    st.sidebar.info("No real API cost in mock mode.")

    uploaded_file = st.file_uploader(
        "Upload a question image",
        type=["png", "jpg", "jpeg"],
    )

    save_path = None
    if uploaded_file is not None:
        uploads_dir = Path("uploads") / "streamlit"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        save_path = uploads_dir / Path(uploaded_file.name).name
        file_bytes = uploaded_file.getvalue()
        save_path.write_bytes(file_bytes)

        st.subheader("Uploaded image")
        st.image(file_bytes, width="stretch")

    st.subheader("Pipeline mode")
    mode = st.selectbox("Select pipeline", ["OCR + LLM"], index=0)
    st.write("Only OCR + LLM is supported in this demo." )

    if st.button("Solve Question"):
        if save_path is None:
            st.error("Please upload an image before solving.")
            return

        with st.spinner("Running OCR + LLM pipeline..."):
            result = solve_question_image(str(save_path), mode="ocr")

        status = result.get("status", "failed")
        if status == "success":
            st.success("Question solved successfully.")
        else:
            st.error("Failed to solve the question.")

        ocr_text = result.get("ocr_result", {}).get("text", "")
        st.subheader("OCR Extracted Text")
        st.code(ocr_text or "(no text extracted)")

        st.subheader("Results")
        st.write(f"**Answer:** {result.get('answer', '')}")
        st.write(f"**Explanation:** {result.get('explanation', '')}")
        st.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")
        st.write(f"**Status:** {status}")

        if result.get("error"):
            st.error(result.get("error"))


if __name__ == "__main__":
    main()
