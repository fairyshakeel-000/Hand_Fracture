import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# -------------------------------
# YOLO CONFIG DIR (deployment-safe)
# -------------------------------
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Hand Fracture Detection 🖐️",
    page_icon="🩻",
    layout="centered"
)

st.title("🖐️ Hand Fracture Detection App")
st.write("Upload a hand X-ray image to detect fractures using your YOLO model.")

MODEL_PATH = r"best.pt"

# -------------------------------
# CHECK REQUIRED LIBRARIES
# -------------------------------
try:
    import numpy as np
except ImportError:
    st.error("""
    ❌ NumPy is not installed!
    This library is required for YOLO predictions.
    
    Fix this by running:
    ```
    pip install numpy
    ```
    """)
    st.stop()

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed:\n\n{e}")
        st.warning("""
        This error is usually caused by PyTorch 2.6+ compatibility.
        Steps to fix:
        1️⃣ Ensure the model file is trusted (your own training).
        2️⃣ Re-export the YOLO model for safe loading:
            yolo export model=best.pt format=pt
        """)
        return None

model = load_model()
if model is None:
    st.stop()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📸 Upload a Hand X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    # Save temporarily for YOLO prediction
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.subheader("🔍 Detecting fractures...")
    try:
        results = model.predict(temp_path)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Detection Result", width='stretch')
        st.success("✅ Detection complete!")

        # -------------------------------
        # SHOW PREDICTION DETAILS
        # -------------------------------
        st.subheader("📊 Prediction Details:")
        if results[0].boxes and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names.get(cls_id, f"Class {cls_id}")

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                width, height = x2 - x1, y2 - y1

                st.markdown(f"""
                **Detection {i + 1}:**
                - 🏷️ Class: `{class_name}`
                - 🎯 Confidence: `{conf:.2f}`
                - 📍 Coordinates:
                    - x1: `{x1:.1f}`, y1: `{y1:.1f}`
                    - x2: `{x2:.1f}`, y2: `{y2:.1f}`
                    - Width: `{width:.1f}`, Height: `{height:.1f}`
                """)
        else:
            st.info("No fractures detected.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Clean up temp file
    os.remove(temp_path)
else:
    st.info("Please upload a hand X-ray image to start detection.")

st.markdown("---")
st.markdown("💡 *Powered by Ultralytics YOLO & Streamlit*")
