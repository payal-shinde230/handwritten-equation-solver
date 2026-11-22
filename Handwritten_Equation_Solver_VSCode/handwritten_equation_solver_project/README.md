# Handwritten Equation Solver

This project implements the approach described in the uploaded report: photo input -> preprocessing -> contour segmentation -> CNN symbol classification -> safe expression evaluation -> Streamlit interface.

## Supported symbols
Digits `0`–`9`, `+`, `-`, `*`, `/`, `(` and `)`.

## 1) Setup
Recommended Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset layout
Add labeled symbol images (cropped images containing one symbol each) under matching folders:

```text
dataset/
  0/  1/  2/ ... 9/
  +/  -/  */  // Note: on Windows, use `multiply` folder instead and adapt label mapping.
  (/  )/
```

The current `train.py` expects exact folder names matching the labels. For a folder named `multiply`, change the folder-to-label mapping in the script or rename it to `*` (the `*` folder name is valid on Windows).

Use a dataset containing both digits and the operators listed above. Do not train on test images used for final evaluation.


## 3) Train the CNN
```bash
python train.py
```
This saves `models/equation_symbol_model.keras` and the best checkpoint.

## 4) Launch the app
```bash
streamlit run app.py
```
The app opens in your browser. Upload a photo or use your webcam, then click **Recognize and Solve**.

## Important limitations
- This baseline assumes one line of arithmetic, with symbols written separately enough for contour segmentation.
- Touching digits may be detected as one contour; symbols with multiple strokes may be split or merged incorrectly.
- The report's example CNN expects a consistent preprocessing style. Dataset and camera images should be similar (white background/dark ink); test preprocessing carefully.
- It currently does not support fractions, exponents, square roots, variables, or multi-line equations.
- The project report's accuracy figures (99.99% training, 99.90% validation, 99.20% held-out symbols) are report results, not guaranteed for this implementation or dataset.
