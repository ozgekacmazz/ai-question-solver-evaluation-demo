# Technical Report

## Project Overview

This project is a demo evaluation framework for solving question images using two distinct approaches:

1. OCR + LLM: extract text from the image and then solve the question with a language model.
2. Direct Vision LLM: send the image directly to a vision-capable language model.

## Problem Definition

The goal is to compare the effectiveness of text-first pipelines against image-first multimodal pipelines for solving question images. The evaluation focuses on accuracy, robustness, and the ability to handle different question formats.

## Architecture

- `app/` contains the API, configuration, and schemas.
- `services/` contains modular pipeline components for preprocessing, OCR, language model interaction, Langflow integration, solving, and evaluation.
- `ui/` contains a Streamlit prototype for future interface development.
- `data/` contains sample questions and ground truth answers.
- `tests/` contains unit tests for core modules.

## Planned Work

- Implement OCR preprocessing with `pytesseract` and OpenCV.
- Connect to an LLM provider for text-based solving.
- Add a multimodal Vision LLM integration for direct image solving.
- Build evaluation metrics and comparison reports.
- Develop a clean Streamlit UI for user workflows.

## Current Status

This repository currently provides a clean project skeleton with placeholder logic. The API includes a health endpoint, configuration supports environment variables, and the service modules are ready for implementation.
