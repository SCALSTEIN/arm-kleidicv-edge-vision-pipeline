import os
from ultralytics import YOLO

def export_yolo():
    output_path = "yolov8n.onnx"
    if not os.path.exists(output_path):
        print("[*] Downloading and exporting YOLOv8n to ONNX format...")
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640, dynamic=False)
        print(f"[✓] Successfully exported model to {output_path}")
    else:
        print(f"[!] {output_path} already exists. Skipping export.")

if __name__ == "__main__":
    export_yolo()
