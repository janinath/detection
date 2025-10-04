import os
import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

# ---------------- YOLO MODEL SETUP ----------------
YOLO_DIR = os.path.join(settings.BASE_DIR, "detection", "yolo")
weights_path = os.path.join(YOLO_DIR, "yolov3-tiny.weights")
config_path = os.path.join(YOLO_DIR, "yolov3-tiny.cfg")
names_path = os.path.join(YOLO_DIR, "coco.names")

net = cv2.dnn.readNet(weights_path, config_path)
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# ---------------- DETECTION FUNCTION ----------------
def yolo_detect(frame, conf_threshold=0.5):
    """Detect objects on a single frame using YOLO"""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                cx, cy, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                x, y = int(cx - w / 2), int(cy - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if indexes is not None and len(indexes) > 0:
        indexes = np.array(indexes).flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {round(confidences[i], 2)}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return frame


# ---------------- PROCESS VIDEO ----------------
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = yolo_detect(frame)
        out.write(processed_frame)

    cap.release()
    out.release()


# ---------------- DJANGO VIEW ----------------
def upload_video(request):
    processed_video_url = None

    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(video_file.name, video_file)
        input_path = os.path.join(upload_dir, filename)

        # Output path
        output_dir = os.path.join(settings.MEDIA_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = "processed_" + filename.split(".")[0] + ".mp4v"
        output_path = os.path.join(output_dir, output_filename)

        # Process video
        process_video(input_path, output_path)

        # ✅ create relative media URL for download
        processed_video_url = f"{settings.MEDIA_URL}outputs/{output_filename}"

    return render(request, "detection/upload.html", {"processed_video_url": processed_video_url})
