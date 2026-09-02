import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import av
import tempfile
import subprocess
import os

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import imageio_ffmpeg


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Traffic Sign Alert System",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦Traffic Sign Alert System")


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

    # Resize to exactly 32 x 32

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Keep the same color format
    # used by OpenCV

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
# VOICE OUTPUT
# ============================================================

def speak_prediction(sign_name):

    speech_text = f"Warning  {sign_name}"

    st.components.v1.html(
        f"""
        <script>
            const text = {speech_text!r};

            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();

                const speech = new SpeechSynthesisUtterance(text);
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
# INPUT TYPE
# ============================================================

input_type = st.radio(
    "Select Input",
    [
        "Image",
        "Video",
        "Live Camera"
    ],
    horizontal=True
)


# ============================================================
# IMAGE INPUT
# ============================================================

if input_type == "Image":

    st.subheader(
        "📷 Upload Traffic Sign Image"
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
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
            # LEFT IMAGE | RIGHT RESULT
            # ==================================================

            col1, col2 = st.columns(2)


            # --------------------------------------------------
            # LEFT SIDE - IMAGE
            # --------------------------------------------------

            with col1:

                st.write(
                    "### Uploaded Image"
                )

                display_original = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    display_original,
                    width=500
                )


            # --------------------------------------------------
            # RIGHT SIDE - PREDICTION
            # --------------------------------------------------

            with col2:

                st.write(
                    "### Prediction"
                )

                st.success(
                    f"Predicted Sign: {sign_name}"
                )

                st.info(
                    f"Confidence: {confidence:.2f}%"
                )

                # Voice output

                speak_prediction(
                    sign_name
                )


# ============================================================
# VIDEO INPUT
# ============================================================

elif input_type == "Video":

    st.subheader(
        "🎥 Upload Traffic Sign Video"
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
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


        # ----------------------------------------------------
        # OPEN VIDEO
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PROCESS VIDEO
        # ----------------------------------------------------

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


            # Progress

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


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    "### Video"
                )

                st.write(
                    "Video processed successfully."
                )


            with col2:

                st.write(
                    "### Prediction"
                )

                st.success(
                    f"Predicted Sign: {sign_name}"
                )

                st.info(
                    f"Confidence: {confidence:.2f}%"
                )

                # Voice output

                speak_prediction(
                    sign_name
                )


        # ----------------------------------------------------
        # CLEAN TEMP FILE
        # ----------------------------------------------------

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

    st.subheader(
        "📹 Live Traffic Sign Prediction"
    )


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
            # No prediction text is drawn on it

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


    st.info(
        "The model input is resized to 32 × 32 pixels."
    )