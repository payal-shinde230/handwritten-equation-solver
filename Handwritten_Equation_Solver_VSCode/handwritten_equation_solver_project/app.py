"""Streamlit app for photo-based handwritten arithmetic."""
import ast
import operator
from pathlib import Path
import re

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras



IMG_SIZE = 45
LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")"]
MODEL_PATH = Path("models/equation_symbol_model.keras")



@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return keras.models.load_model(MODEL_PATH)



def preprocess_for_model(crop):
    """Convert a symbol crop to the CNN's white-on-black 45x45 input."""
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    if crop.mean() > 127:
        crop = 255 - crop
    return crop.astype(np.float32) / 255.0



def merge_boxes(boxes, x_overlap_ratio=0.35):
    """Merge vertically separated strokes of symbols (e.g. '=' and '÷').
    Merge only boxes with substantial horizontal overlap; this is a heuristic.
    """
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    for box in boxes:
        x, y, w, h = box
        changed = True
        while changed:
            changed = False
            for i, (mx, my, mw, mh) in enumerate(merged):
                overlap = max(0, min(x + w, mx + mw) - max(x, mx))
                smaller_width = max(1, min(w, mw))
                if overlap / smaller_width >= x_overlap_ratio:
                    nx, ny = min(x, mx), min(y, my)
                    nr, nb = max(x + w, mx + mw), max(y + h, my + mh)
                    x, y, w, h = nx, ny, nr - nx, nb - ny
                    merged.pop(i)
                    changed = True
                    break
        merged.append((x, y, w, h))
    return sorted(merged, key=lambda b: b[0])


def segment_image(pil_image):
    image = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 20]
    boxes = merge_boxes(boxes)
    symbols = []
    for x, y, w, h in boxes:
        crop = threshold[y:y+h, x:x+w]
        side = max(w, h) + 10
        canvas = np.zeros((side, side), dtype=np.uint8)
        yoff, xoff = (side - h) // 2, (side - w) // 2
        canvas[yoff:yoff+h, xoff:xoff+w] = crop
        symbols.append(preprocess_for_model(canvas))
    return np.array(symbols)[..., None], boxes, threshold



def safe_calculate(expression):
    """Safely evaluate numeric arithmetic only; never use Python eval()."""
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        raise ValueError("Expression contains unsupported characters")
    tree = ast.parse(expression, mode="eval")
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](visit(node.operand))
        raise ValueError("Unsupported expression syntax")
    return visit(tree)


st.set_page_config(page_title="Handwritten Equation Solver", page_icon="🧮", layout="centered")
st.title("🧮 Handwritten Equation Solver")
st.write("Upload or capture a handwritten single-line arithmetic equation.")
model = load_model()
if model is None:
    st.warning("CNN model not found yet. First prepare the dataset and run `python train.py`.")

uploaded = st.file_uploader("Upload an equation image", type=["png", "jpg", "jpeg"])
camera = st.camera_input("Or take a photo")
image_source = camera if camera is not None else uploaded
if image_source is not None:
    pil_img = Image.open(image_source).convert("RGB")
    st.image(pil_img, caption="Input image", use_container_width=True)
    if st.button("Recognize and Solve", type="primary", disabled=(model is None)):
        with st.spinner("Recognizing symbols..."):
            symbol_images, boxes, binary = segment_image(pil_img)
            if len(symbol_images) == 0:
                st.error("No symbols detected. Try a clearer, well-lit image.")
            else:
                probabilities = model.predict(symbol_images, verbose=0)
                indices = probabilities.argmax(axis=1)
                expression = "".join(LABELS[i] for i in indices)
                confidence = probabilities.max(axis=1)
                st.subheader("Recognized expression")
                st.code(expression)
                st.caption("Symbol confidence: " + ", ".join(f"{c:.0%}" for c in confidence))
                try:
                    result = safe_calculate(expression)
                    st.success(f"Answer: {result}")
                except Exception as exc:
                    st.warning(f"Could not evaluate expression: {exc}")
                st.image(binary, caption="Thresholded image used for segmentation")
                st.write("Detected symbols (left to right):", len(boxes))
