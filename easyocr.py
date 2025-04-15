import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
import easyocr

st.set_page_config(page_title="Kannada + English OCR", layout="centered")  # <- FIRST Streamlit command

OCR_LANGUAGES = ['en', 'kn']  # English + Kannada

st.title("📄 Kannada + English OCR App")
st.caption("Streamlit + EasyOCR (no Tesseract needed!)")


# --- Preprocessing function ---
def preprocess_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    img = cv2.fastNlMeansDenoising(img, h=30)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    return Image.fromarray(img)

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload PDF or Image (JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        st.image(doc[0].get_pixmap(dpi=150).tobytes("png"), caption="First Page Preview", use_container_width=True)
        images = []

        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes))
            images.append(pil_img)
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        images = [img]

    st.info(f"🧠 Processing {len(images)} page(s) with Kannada + English OCR...")

    # Initialize EasyOCR
    reader = easyocr.Reader(OCR_LANGUAGES)

    all_text = ""
    for i, img in enumerate(images):
        st.write(f"🔍 OCR on Page {i + 1}...")
        clean_img = preprocess_image(img)
        result = reader.readtext(np.array(clean_img))

        extracted_text = "\n".join([line[1] for line in result])
        all_text += f"\n--- Page {i+1} ---\n" + extracted_text

    st.success("✅ OCR Completed")
    st.download_button("📥 Download Extracted Text", all_text, file_name="extracted_text.txt")
    st.text_area("📄 Extracted Text Preview", all_text, height=300)
