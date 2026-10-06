import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from io import BytesIO


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartPark AI",
    page_icon="🅿️",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-top: 0;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<p class="main-title">🅿️ SmartPark AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Intelligent Parking Detection System using YOLO and OpenCV</p>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Loads the YOLO model once and keeps it cached.
    YOLO11n is a lightweight model suitable for local testing.
    """
    return YOLO("yolo11n.pt")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the YOLO model.")
    st.error(str(e))
    st.info(
        "Make sure you have an internet connection during the first run "
        "so Ultralytics can download the model."
    )
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05
)

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

selected_classes = st.sidebar.multiselect(
    "Vehicle Types",
    options=list(vehicle_classes.values()),
    default=["Car", "Motorcycle", "Bus", "Truck"]
)

st.sidebar.divider()

st.sidebar.subheader("🅿️ Parking Zone")

st.sidebar.write(
    "Optional: define a parking zone using normalized coordinates."
)

st.sidebar.caption(
    "Format: x1,y1,x2,y2 as percentages of image width/height."
)

zone_text = st.sidebar.text_input(
    "Zone Coordinates",
    value="10,10,90,90"
)

show_labels = st.sidebar.checkbox(
    "Show Vehicle Labels",
    value=True
)

show_boxes = st.sidebar.checkbox(
    "Show Bounding Boxes",
    value=True
)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def parse_zone(zone_text, width, height):
    """
    Converts normalized percentage coordinates into pixel coordinates.
    """

    try:
        values = [float(x.strip()) for x in zone_text.split(",")]

        if len(values) != 4:
            return None

        x1, y1, x2, y2 = values

        x1 = int(width * x1 / 100)
        y1 = int(height * y1 / 100)
        x2 = int(width * x2 / 100)
        y2 = int(height * y2 / 100)

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        return x1, y1, x2, y2

    except Exception:
        return None


def calculate_center(x1, y1, x2, y2):
    """Returns the center point of a bounding box."""

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


def process_frame(frame):
    """
    Runs YOLO detection on a frame and returns:
    processed frame, vehicle count, vehicle information.
    """

    height, width = frame.shape[:2]

    results = model.predict(
        source=frame,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    processed_frame = frame.copy()

    vehicle_count = 0
    occupied_count = 0

    detected_vehicles = []

    zone = parse_zone(zone_text, width, height)

    # Draw parking zone
    if zone is not None:
        zx1, zy1, zx2, zy2 = zone

        cv2.rectangle(
            processed_frame,
            (zx1, zy1),
            (zx2, zy2),
            (255, 180, 0),
            2
        )

        cv2.putText(
            processed_frame,
            "PARKING ZONE",
            (zx1, max(25, zy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 0),
            2
        )

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])

            if class_id not in vehicle_classes:
                continue

            vehicle_name = vehicle_classes[class_id]

            if vehicle_name not in selected_classes:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            vehicle_count += 1

            center_x, center_y = calculate_center(
                x1, y1, x2, y2
            )

            inside_zone = False

            if zone is not None:
                zx1, zy1, zx2, zy2 = zone

                if (
                    zx1 <= center_x <= zx2
                    and
                    zy1 <= center_y <= zy2
                ):
                    inside_zone = True
                    occupied_count += 1

            detected_vehicles.append(
                {
                    "type": vehicle_name,
                    "confidence": confidence_score,
                    "center": (center_x, center_y),
                    "inside_zone": inside_zone
                }
            )

            # -------------------------------------------------
            # DRAW BOUNDING BOX
            # -------------------------------------------------
            if show_boxes:

                if inside_zone:
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(
                    processed_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

            # -------------------------------------------------
            # DRAW LABEL
            # -------------------------------------------------
            if show_labels:

                label = (
                    f"{vehicle_name} "
                    f"{confidence_score:.0%}"
                )

                cv2.putText(
                    processed_frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

    return (
        processed_frame,
        vehicle_count,
        occupied_count,
        detected_vehicles
    )


# ---------------------------------------------------------
# INPUT TYPE
# ---------------------------------------------------------
st.subheader("📂 Select Input")

input_type = st.radio(
    "Choose how you want to test SmartPark AI:",
    ["Image", "Video"],
    horizontal=True
)


# =========================================================
# IMAGE MODE
# =========================================================
if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload a parking image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        image_array = np.array(image)

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        if st.button(
            "🔍 Detect Vehicles",
            type="primary"
        ):

            with st.spinner("Analyzing parking area..."):

                processed_frame, vehicle_count, occupied_count, vehicles = (
                    process_frame(frame)
                )

            processed_rgb = cv2.cvtColor(
                processed_frame,
                cv2.COLOR_BGR2RGB
            )

            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🚗 Vehicles Detected",
                    vehicle_count
                )

            with col2:
                st.metric(
                    "🅿️ Vehicles in Zone",
                    occupied_count
                )

            with col3:

                if occupied_count > 0:
                    status = "Occupied"
                else:
                    status = "Available"

                st.metric(
                    "📍 Parking Status",
                    status
                )

            st.divider()

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------
            st.subheader("📸 Detection Result")

            st.image(
                processed_rgb,
                use_container_width=True
            )

            # -------------------------------------------------
            # DOWNLOAD RESULT
            # -------------------------------------------------
            output_image = Image.fromarray(
                processed_rgb
            )

            image_buffer = BytesIO()

            output_image.save(
                image_buffer,
                format="PNG"
            )

            st.download_button(
                label="⬇️ Download Result",
                data=image_buffer.getvalue(),
                file_name="smartpark_detection.png",
                mime="image/png"
            )

            # -------------------------------------------------
            # DETECTION SUMMARY
            # -------------------------------------------------
            st.subheader("📊 Detection Summary")

            if vehicles:

                for index, vehicle in enumerate(
                    vehicles,
                    start=1
                ):

                    location = (
                        "Inside Parking Zone"
                        if vehicle["inside_zone"]
                        else "Outside Parking Zone"
                    )

                    st.write(
                        f"**{index}. {vehicle['type']}** — "
                        f"{vehicle['confidence']:.1%} confidence — "
                        f"{location}"
                    )

            else:
                st.info("No supported vehicles detected.")


# =========================================================
# VIDEO MODE
# =========================================================
else:

    uploaded_video = st.file_uploader(
        "Upload a parking video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        video_bytes = uploaded_video.read()

        temporary_video = "temp_input_video.mp4"

        with open(
            temporary_video,
            "wb"
        ) as video_file:

            video_file.write(video_bytes)

        if st.button(
            "🎥 Analyze Video",
            type="primary"
        ):

            cap = cv2.VideoCapture(
                temporary_video
            )

            if not cap.isOpened():

                st.error(
                    "Unable to open the uploaded video."
                )

                st.stop()

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:
                fps = 25

            frame_width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            frame_height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            st.subheader(
                "🎬 Video Analysis"
            )

            video_placeholder = st.empty()
            progress_bar = st.progress(0)

            frame_number = 0

            max_vehicle_count = 0
            max_occupied_count = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                # Process every frame
                (
                    processed_frame,
                    vehicle_count,
                    occupied_count,
                    _
                ) = process_frame(frame)

                max_vehicle_count = max(
                    max_vehicle_count,
                    vehicle_count
                )

                max_occupied_count = max(
                    max_occupied_count,
                    occupied_count
                )

                display_frame = cv2.cvtColor(
                    processed_frame,
                    cv2.COLOR_BGR2RGB
                )

                video_placeholder.image(
                    display_frame,
                    channels="RGB",
                    use_container_width=True
                )

                if total_frames > 0:

                    progress = (
                        frame_number /
                        total_frames
                    )

                    progress_bar.progress(
                        min(progress, 1.0)
                    )

            cap.release()

            progress_bar.progress(1.0)

            st.success(
                "Video analysis completed successfully."
            )

            # -------------------------------------------------
            # VIDEO SUMMARY
            # -------------------------------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Maximum Vehicles Detected",
                    max_vehicle_count
                )

            with col2:
                st.metric(
                    "Maximum Zone Occupancy",
                    max_occupied_count
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "SmartPark AI • Intelligent Parking Detection System "
    "• Built with Streamlit, OpenCV and Ultralytics YOLO"
)