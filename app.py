import os
import time
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import streamlit as st

st.set_page_config(
    page_title="Arm Edge Vision Telemetry",
    page_icon="👁️",
    layout="wide"
)

# Standard 80-Class COCO Labels
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_PATH = "yolov8n.onnx"
MODEL_URLS = [
    "https://huggingface.co/visual-layer/yolov8n-onnx/resolve/main/yolov8n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
]

def ensure_model_exists():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return

    with st.spinner("Downloading YOLOv8n ONNX model (~12MB)..."):
        success = False
        for url in MODEL_URLS:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp, open(MODEL_PATH, "wb") as out_file:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)

                if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
                    success = True
                    break
            except Exception:
                continue

        if not success:
            st.error("Failed to load model weights. Ensure 'yolov8n.onnx' is committed directly to the repository.")
            st.stop()

ensure_model_exists()

@st.cache_resource(show_spinner="Initializing ONNX Runtime Engine...")
def load_session():
    opts = ort.SessionOptions()
    cpu_cores = os.cpu_count() or 4
    opts.intra_op_num_threads = min(4, cpu_cores)
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=['CPUExecutionProvider']
    )
    return session

session = load_session()
input_name = session.get_inputs()[0].name

# ----------------- Pipeline Functions -----------------
def preprocess(frame, target_size=(640, 640)):
    """Vectorized preprocessing: Resize -> BGR2RGB -> Transpose -> Normalize."""
    t0 = time.perf_counter()

    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    t_pre = (time.perf_counter() - t0) * 1000.0
    return tensor, t_pre

def postprocess(output, orig_shape, conf_threshold=0.35):
    """Parse YOLOv8 output tensor and extract non-suppressed bounding boxes."""
    t0 = time.perf_counter()
    predictions = np.squeeze(output[0]).T

    scores = np.max(predictions[:, 4:], axis=1)
    keep = scores > conf_threshold
    predictions = predictions[keep]
    scores = scores[keep]

    boxes = []
    if len(scores) > 0:
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        x, y, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]

        scale_x = orig_shape[1] / 640.0
        scale_y = orig_shape[0] / 640.0

        for i in range(len(scores)):
            x1 = int((x[i] - w[i] / 2) * scale_x)
            y1 = int((y[i] - h[i] / 2) * scale_y)
            x2 = int((x[i] + w[i] / 2) * scale_x)
            y2 = int((y[i] + h[i] / 2) * scale_y)
            boxes.append((x1, y1, x2, y2, scores[i], class_ids[i]))

    t_post = (time.perf_counter() - t0) * 1000.0
    return boxes, t_post

# ----------------- Sidebar Telemetry -----------------
st.sidebar.title("📊 Silicon Telemetry")
col_cpu, col_ram = st.sidebar.columns(2)
cpu_metric = col_cpu.empty()
ram_metric = col_ram.empty()

st.sidebar.markdown("### Frame-Time Breakdown (ms)")
latency_chart = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.markdown("**Engine:** ONNX Runtime (`CPUExecutionProvider`)")
st.sidebar.markdown("**Acceleration:** Arm Neon / KleidiCV Vector Routines")
st.sidebar.markdown("**Detector:** YOLOv8 Nano (640x640)")

# ----------------- Main Console -----------------
st.title("👁️ Edge Vision Pipeline: Arm Preprocessing & YOLOv8")
st.caption("Profiling frame preprocessing reduction and neural inference throughput on CPU silicon.")

source_type = st.radio("Input Source Selection:", ["Sample Test Image", "Webcam Live Feed"], horizontal=True)

if "latency_history" not in st.session_state:
    st.session_state.latency_history = pd.DataFrame(
        columns=["Frame", "Preprocessing (ms)", "Inference (ms)"]
    )

frame_display = st.empty()
m1, m2, m3, m4 = st.columns(4)

def process_and_render(frame):
    h, w, _ = frame.shape
    tensor, t_pre = preprocess(frame)

    # Inference
    t_inf_start = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    t_inf = (time.perf_counter() - t_inf_start) * 1000.0

    # Postprocessing
    boxes, t_post = postprocess(outputs, (h, w))

    # Render Detections
    annotated = frame.copy()
    for (x1, y1, x2, y2, conf, cls_id) in boxes:
        label = f"{CLASSES[cls_id]}: {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    total_time = t_pre + t_inf + t_post
    fps = 1000.0 / total_time if total_time > 0 else 0.0

    frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    m1.metric("Pipeline FPS", f"{fps:.1f} FPS")
    m2.metric("Preprocessing", f"{t_pre:.2f} ms")
    m3.metric("Inference", f"{t_inf:.2f} ms")
    m4.metric("Postprocessing", f"{t_post:.2f} ms")

    new_entry = pd.DataFrame([{
        "Frame": len(st.session_state.latency_history) + 1,
        "Preprocessing (ms)": round(t_pre, 2),
        "Inference (ms)": round(t_inf, 2)
    }])
    st.session_state.latency_history = pd.concat(
        [st.session_state.latency_history, new_entry],
        ignore_index=True
    ).tail(30)

    latency_chart.line_chart(
        st.session_state.latency_history.set_index("Frame")[["Preprocessing (ms)", "Inference (ms)"]]
    )

    ram = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=None)
    cpu_metric.metric("CPU Load", f"{cpu_pct}%")
    ram_metric.metric("RAM Used", f"{round(ram.used / (1024**3), 2)} GB")

# ----------------- Execution Logic -----------------
if source_type == "Sample Test Image":
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.circle(img, (640, 360), 140, (0, 180, 255), -1)
    cv2.rectangle(img, (220, 220), (460, 560), (255, 120, 0), -1)
    cv2.putText(img, "Synthetic Test Frame (Ready for Camera Feed)", (140, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if st.button("▶ Run Single Frame Benchmark"):
        process_and_render(img)

elif source_type == "Webcam Live Feed":
    run_cam = st.toggle("Start Camera Stream", value=False)
    cap = cv2.VideoCapture(0)

    while run_cam and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to acquire camera feed.")
            break
        process_and_render(frame)
        time.sleep(0.01)

    cap.release()
