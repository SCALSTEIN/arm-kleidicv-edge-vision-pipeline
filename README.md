# Low-Latency Edge Vision Pipeline: Arm KleidiCV & YOLOv8 Optimization

A high-throughput edge computer vision pipeline designed to eliminate image preprocessing bottlenecks (color conversion, bilinear interpolation, normalization) before feeding frames to an on-device neural detector.

## Key Features
- **Hardware-Aware Preprocessing:** Analyzes frame transformation latency using vectorized SIMD (Arm Neon / KleidiCV concepts).
- **Embedded ONNX Engine:** Runs multi-threaded YOLOv8-Nano on CPU silicon with zero external GPU dependencies.
- **Interactive Telemetry Dashboard:** Live Streamlit UI displaying pipeline FPS, per-frame latency breakdown (ms), and CPU/RAM consumption curves.

## Quickstart

```bash
# 1. Clone repo
git clone [https://github.com/SCALSTEIN/arm-kleidicv-edge-vision-pipeline.git](https://github.com/SCALSTEIN/arm-kleidicv-edge-vision-pipeline.git)
cd arm-kleidicv-edge-vision-pipeline

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch console
streamlit run app.py
