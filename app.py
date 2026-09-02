import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import av
import tempfile
import os

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* Main application */

.stApp {
    background: linear-gradient(
        135deg,
        #f5f7fa 0%,
        #e9eef5 100%
    );
}


/* Page spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* Main title */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #292d3d;
    margin-bottom: 5px;
}


/* Subtitle */

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #667085;
    margin-bottom: 30px;
}


/* Feature cards */

.feature-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    min-height: 145px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.06);
}


.feature-icon {
    font-size: 34px;
}


.feature-title {
    font-size: 18px;
    font-weight: 700;
    color: #292d3d;
    margin-top: 8px;
}


.feature-text {
    font-size: 14px;
    color: #667085;
    margin-top: 6px;
}


/* Result box */

.result-box {
    background: white;
    border: 2px solid #ff4b4b;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.07);
}


/* Prediction name */

.prediction-name {
    font-size: 30px;
    font-weight: 800;
    color: #292d3d;
    margin: 10px 0;
}


/* Confidence */

.confidence-number {
    font-size: 27px;
    font-weight: 800;
    color: #ff4b4b;
}


/* Information cards */

.info-box {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}


/* Footer */

.footer {
    text-align: center;
    color: #667085;
    font-size: 13px;
    padding: 25px;
    margin-top: 40px;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background-color: white;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
🚨 Smart Traffic Sign Recognition
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
AI-powered traffic sign detection, classification & voice warning system
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/vehicle_sign_ann.keras"
CSV_PATH = "data/traffic_sign.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ============================================================
# LOAD SIGN MAPPING
# ============================================================

@st.cache_data
def load_sign_mapping():

    df = pd.read_csv(
        CSV_PATH
    )

    return dict(
        zip(
            df["ClassId"],
            df["Name"]
        )
    )


sign_mapping = load_sign_mapping()


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = (32, 32)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    # Resize

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Convert to float

    image = image.astype(
        np.float32
    )

    # Normalize

    image = image / 255.0

    # Flatten

    image = image.reshape(
        1,
        -1
    )

    return image


# ============================================================
# PREDICT
# ============================================================

def predict(image):

    processed = preprocess_image(
        image
    )

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    class_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        np.max(probabilities) * 100
    )

    sign_name = sign_mapping.get(
        class_id,
        "Unknown Sign"
    )

    return (
        class_id,
        sign_name,
        confidence
    )


# ============================================================
# VOICE ALERT
# ============================================================

def speak_prediction(sign_name):

    speech_text = f"Warning {sign_name}"

    st.components.v1.html(
        f"""
        <script>

        const text = {speech_text!r};

        if ("speechSynthesis" in window) {{

            window.speechSynthesis.cancel();

            const speech =
                new SpeechSynthesisUtterance(text);

            speech.rate = 0.9;
            speech.pitch = 1.0;
            speech.volume = 1.0;

            window.speechSynthesis.speak(speech);
        }}

        </script>
        """,
        height=0
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚦 Traffic Sign AI")

    st.write(
        "Choose an input method below."
    )

    st.divider()

    input_type = st.radio(
        "Input Mode",
        [
            "📷 Image",
            "🎥 Video",
            "📹 Live Camera"
        ]
    )

    st.divider()

    st.markdown("### 🧠 Model Information")

    st.write(
        "Model: Artificial Neural Network"
    )

    st.write(
        "Input Size: 32 × 32"
    )

    st.write(
        "Input Type: Image Pixels"
    )

    st.divider()

    st.success(
        "🟢 System Ready"
    )


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">📷</div>
<div class="feature-title">Image Recognition</div>
<div class="feature-text">Upload a traffic sign image</div>
</div>
""",
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🎥</div>
<div class="feature-title">Video Analysis</div>
<div class="feature-text">Analyze traffic sign videos</div>
</div>
""",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🔊</div>
<div class="feature-title">Voice Warning</div>
<div class="feature-text">Hear alerts for detected signs</div>
</div>
""",
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# IMAGE MODE
# ============================================================

if input_type == "📷 Image":

    st.markdown(
        "## 📷 Traffic Sign Image Analysis"
    )

    st.caption(
        "Upload an image to identify the traffic sign."
    )

    uploaded_image = st.file_uploader(
        "Choose a traffic sign image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_image:

        file_bytes = np.asarray(
            bytearray(
                uploaded_image.read()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "Unable to read image."
            )

        else:

            # ==================================================
            # PREDICTION
            # ==================================================

            class_id, sign_name, confidence = predict(
                image
            )


            # ==================================================
            # IMAGE + RESULT
            # ==================================================

            left, right = st.columns(
                [1.15, 1]
            )


            # --------------------------------------------------
            # LEFT
            # --------------------------------------------------

            with left:

                st.markdown(
                    "### 🖼️ Input Image"
                )

                display_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    display_image,
                    width=400
                )


            # --------------------------------------------------
            # RIGHT
            # --------------------------------------------------

            with right:

                st.markdown(
                    "### 🚨 Detection Result"
                )

                # Native Streamlit container
                # No nested HTML

                with st.container(
                    border=True
                ):

                    st.caption(
                        "Predicted Traffic Sign"
                    )

                    st.markdown(
                        f"""
### 🚦 {sign_name}
"""
                    )

                    st.caption(
                        "Confidence Score"
                    )

                    st.markdown(
                        f"## {confidence:.2f}%"
                    )

                    st.progress(
                        min(
                            confidence / 100,
                            1.0
                        )
                    )


                st.write("")


                # Voice alert

                st.success(
                    "🔊 Voice alert generated"
                )

                speak_prediction(
                    sign_name
                )


           
# ============================================================
# VIDEO MODE
# ============================================================

elif input_type == "🎥 Video":

    st.markdown(
        "## 🎥 Traffic Sign Video Analysis"
    )

    st.caption(
        "Upload a video containing traffic signs."
    )


    uploaded_video = st.file_uploader(
        "Choose a traffic sign video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )


    if uploaded_video:

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        input_path = input_file.name


        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            input_path
        )


        if not cap.isOpened():

            st.error(
                "Unable to open video."
            )

            st.stop()


        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        info1, info2, info3 = st.columns(3)


        with info1:

            st.metric(
                "🎬 Total Frames",
                f"{total_frames:,}"
            )


        with info2:

            st.metric(
                "🧠 Model",
                "ANN"
            )


        with info3:

            st.metric(
                "📐 Input",
                "32 × 32"
            )


        st.write("")


        # ====================================================
        # PROCESS
        # ====================================================

        st.markdown(
            "### 🔄 Processing Video"
        )


        progress = st.progress(
            0
        )


        frame_count = 0

        first_prediction = None


        while True:

            ret, frame = cap.read()


            if not ret:

                break


            # Prediction

            class_id, sign_name, confidence = predict(
                frame
            )


            # Save first prediction

            if first_prediction is None:

                first_prediction = (
                    class_id,
                    sign_name,
                    confidence
                )


            frame_count += 1


            if total_frames > 0:

                progress.progress(
                    min(
                        frame_count / total_frames,
                        1.0
                    )
                )


        cap.release()

        progress.empty()


        # ====================================================
        # RESULT
        # ====================================================

        if first_prediction is not None:

            class_id, sign_name, confidence = first_prediction


            st.success(
                "✅ Video processed successfully"
            )


            result1, result2 = st.columns(2)


            with result1:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### 🎬 Processing Complete"
                    )

                    st.metric(
                        "Frames Processed",
                        f"{frame_count:,}"
                    )


            with result2:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### 🚨 Prediction"
                    )

                    st.markdown(
                        f"## 🚦 {sign_name}"
                    )

                    st.write(
                        f"Confidence: **{confidence:.2f}%**"
                    )

                    st.progress(
                        min(
                            confidence / 100,
                            1.0
                        )
                    )


            st.write("")

            speak_prediction(
                sign_name
            )


        # ====================================================
        # DELETE TEMP FILE
        # ====================================================

        try:

            os.remove(
                input_path
            )

        except:

            pass


# ============================================================
# LIVE CAMERA
# ============================================================

else:

    st.markdown(
        "## 📹 Live Traffic Sign Recognition"
    )

    st.caption(
        "Start your camera to process traffic signs in real time."
    )


    st.info(
        "📸 Allow camera access when your browser asks for permission."
    )


    # ========================================================
    # VIDEO PROCESSOR
    # ========================================================

    class VideoProcessor(
        VideoProcessorBase
    ):

        def recv(
            self,
            frame
        ):

            # Get camera frame

            image = frame.to_ndarray(
                format="bgr24"
            )


            # Prediction

            class_id, sign_name, confidence = predict(
                image
            )


            # Keep original camera frame

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )


    # ========================================================
    # WEBRTC
    # ========================================================

    webrtc_streamer(

        key="vehicle-sign-camera",

        video_processor_factory=VideoProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    st.write("")


    # ========================================================
    # CAMERA INFORMATION
    # ========================================================

    info1, info2, info3 = st.columns(3)


    with info1:

        with st.container(
            border=True
        ):

            st.metric(
                "📹 Input",
                "Live Camera"
            )


    with info2:

        with st.container(
            border=True
        ):

            st.metric(
                "🧠 Model",
                "ANN"
            )


    with info3:

        with st.container(
            border=True
        ):

            st.metric(
                "📐 Resolution",
                "32 × 32"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
🚦 <b>Smart Traffic Sign Recognition & Alert System</b>
<br><br>
Powered by Artificial Neural Networks • Computer Vision • Streamlit
</div>
""",
    unsafe_allow_html=True
)