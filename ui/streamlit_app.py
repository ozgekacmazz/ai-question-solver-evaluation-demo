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
        "Mock mode is enabled by default, so you can try the flow without API keys."
    )

    st.sidebar.header("Project status")
    st.sidebar.write("OCR preprocessing pipeline works.")
    st.sidebar.write("Available modes: OCR, Vision, Compare, Adaptive Auto, and Langflow")
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
    mode_label = st.selectbox(
        "Select pipeline",
        ["OCR + LLM", "Direct Vision LLM", "Both / Compare", "Adaptive Auto", "OCR + Langflow"],
        index=0,
    )
    mode_map = {
        "OCR + LLM": "ocr",
        "Direct Vision LLM": "vision",
        "Both / Compare": "both",
        "Adaptive Auto": "adaptive",
        "OCR + Langflow": "ocr_langflow",
    }
    mode = mode_map[mode_label]

    if st.button("Solve Question"):
        if save_path is None:
            st.error("Please upload an image before solving.")
            return

        with st.spinner(f"Running {mode_label} pipeline..."):
            result = solve_question_image(str(save_path), mode=mode)

        status = result.get("status", "failed")
        if status == "success":
            st.success("Question solved successfully.")
        else:
            st.error("Failed to solve the question.")

        router_decision = result.get("router_decision")
        if result.get("adaptive_selected_mode") or isinstance(router_decision, dict):
            st.subheader("Adaptive Router Decision")
            st.write(f"**Selected mode:** {result.get('adaptive_selected_mode', '')}")
            if isinstance(router_decision, dict):
                st.write(f"**Detected subject:** {router_decision.get('detected_subject', '')}")
                st.write(f"**Detected question type:** {router_decision.get('detected_question_type', '')}")
                st.write(f"**Recommended mode:** {router_decision.get('recommended_mode', '')}")
                st.write(f"**Reason:** {router_decision.get('reason', '')}")
                router_confidence = router_decision.get("confidence")
                if router_confidence is not None:
                    st.metric("Router confidence", f"{router_confidence:.2f}")

        if result.get("langflow_status") or result.get("langflow_error"):
            st.subheader("Langflow Status")
            if result.get("langflow_status"):
                st.write(f"**Status:** {result.get('langflow_status')}")
            if result.get("langflow_error"):
                st.write(f"**Error:** {result.get('langflow_error')}")

        display_mode = result.get("adaptive_selected_mode") if mode == "adaptive" else mode

        if display_mode in {"ocr", "ocr_langflow"}:
            ocr_text = result.get("ocr_result", {}).get("text", "")
            st.subheader("OCR Extracted Text")
            st.code(ocr_text or "(no text extracted)")

            st.subheader("Results")
            st.write(f"**Answer:** {result.get('answer', '')}")
            st.write(f"**Explanation:** {result.get('explanation', '')}")
            if result.get("provider_mode"):
                st.write(f"**Provider mode:** {result.get('provider_mode')}")
            st.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")

        elif display_mode == "vision":
            st.subheader("Vision Result")
            st.write(f"**Answer:** {result.get('answer', '')}")
            st.write(f"**Explanation:** {result.get('explanation', '')}")
            if result.get("provider_mode"):
                st.write(f"**Provider mode:** {result.get('provider_mode')}")
            st.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")

        else:
            ocr_result = result.get("ocr_pipeline_result", {})
            vision_result = result.get("vision_pipeline_result", {})

            st.subheader("OCR + LLM Result")
            st.code(ocr_result.get("ocr_result", {}).get("text", "") or "(no text extracted)")
            st.write(f"**Answer:** {ocr_result.get('answer', '')}")
            st.write(f"**Explanation:** {ocr_result.get('explanation', '')}")
            if result.get("ocr_provider_mode") or ocr_result.get("provider_mode"):
                st.write(f"**OCR provider mode:** {result.get('ocr_provider_mode') or ocr_result.get('provider_mode')}")
            st.metric("OCR Confidence", f"{ocr_result.get('confidence', 0.0):.2f}")

            st.subheader("Vision LLM Result")
            st.write(f"**Answer:** {vision_result.get('answer', '')}")
            st.write(f"**Explanation:** {vision_result.get('explanation', '')}")
            if result.get("vision_provider_mode") or vision_result.get("provider_mode"):
                st.write(f"**Vision provider mode:** {result.get('vision_provider_mode') or vision_result.get('provider_mode')}")
            st.metric("Vision Confidence", f"{vision_result.get('confidence', 0.0):.2f}")

            st.subheader("Recommendation")
            st.write(f"**Recommended pipeline:** {result.get('recommended_pipeline', '')}")
            st.write(result.get("comparison_summary", ""))
            st.write(f"**Final answer:** {result.get('answer', '')}")
            st.metric("Recommended Confidence", f"{result.get('confidence', 0.0):.2f}")

        st.write(f"**Status:** {status}")

        if result.get("error"):
            st.error(result.get("error"))


if __name__ == "__main__":
    main()
