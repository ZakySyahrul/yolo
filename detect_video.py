import cv2
import numpy as np
import pandas as pd
import random
from ultralytics import YOLO
from augmentation import random_augmentations
from sklearn.cluster import KMeans


def _safe_get_cls(box):
    """Extract YOLO class ID safely."""
    try:
        val = box.cls
        if hasattr(val, "__len__") and len(val) > 0:
            try:
                return int(val[0].item())
            except:
                return int(val[0])
        return int(val)
    except:
        return -1



# NEW DENSITY CLASSIFICATION (k-means + fallback threshold)


def classify_density_kmeans(total_count, history):
    """
    Adaptive traffic density using KMeans (3 clusters):
    - cluster 0 = Lancar
    - cluster 1 = Ramai
    - cluster 2 = Padat
    
    If dataset too small / KMeans fails → fallback threshold.
    """
    history = history[-300:]  # limit memory for speed
    
    # minimal data for KMeans
    if len(history) < 15:
        if total_count < 10:
            return "Lancar"
        elif total_count < 25:
            return "Ramai"
        return "Padat"
    
    try:
        X = np.array(history).reshape(-1, 1)
        km = KMeans(n_clusters=3, n_init="auto")
        km.fit(X)

        centers = sorted([(i, c[0]) for i, c in enumerate(km.cluster_centers_)],
                         key=lambda x: x[1])  

        mapping = {
            centers[0][0]: "Lancar",
            centers[1][0]: "Ramai",
            centers[2][0]: "Padat"
        }

        label = km.predict(np.array([[total_count]]))[0]
        return mapping[label]

    except:
        # fallback threshold
        if total_count < 10:
            return "Lancar"
        elif total_count < 25:
            return "Ramai"
        return "Padat"


# MAIN VIDEO PROCESSOR

def process_video(
    model,
    input_video,
    output_path="output.mp4",
    sample_count=6,
    conf=0.5
):

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    df_records = []
    frame_id = 0
    sample_frames = []
    history_counts = []

    # Sampling frames untuk augmentation
    if total_frames <= 0:
        sample_indexes = []
    else:
        num_samples = min(sample_count, max(1, total_frames))
        if total_frames >= num_samples:
            sample_indexes = sorted(random.sample(range(total_frames), num_samples))
        else:
            sample_indexes = list(range(total_frames))

    # LOOP FRAME

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in sample_indexes:
            sample_frames.append(frame.copy())

        # Time stamps
        second = frame_id / fps
        minute = second / 60
        hour   = second / 3600

        # YOLO prediction
        results = model.predict(frame, conf=conf, verbose=False)
        if not results:
            boxes, names_local = [], {}
        else:
            boxes = results[0].boxes
            names_local = results[0].names

        # Count by type
        count_by_type = {}
        for box in boxes:
            cls_id = _safe_get_cls(box)
            if cls_id < 0:
                continue

            label = names_local.get(cls_id, f"class_{cls_id}")
            count_by_type[label] = count_by_type.get(label, 0) + 1

        total = sum(count_by_type.values())
        history_counts.append(total)

        # Density using KMeans
        density = classify_density_kmeans(total, history_counts)

        # Record dict
        record = {
            "frame": frame_id,
            "second": second,
            "minute": minute,
            "hour": hour,
            "total": total,
            "density": density
        }

        # Tambahkan type_count
        for k, v in count_by_type.items():
            record[f"type_{k}"] = v

        df_records.append(record)

        # Render video
        try:
            annotated = results[0].plot()
            if annotated is None:
                out.write(frame)
            else:
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                out.write(annotated_bgr)
        except:
            out.write(frame)

        frame_id += 1

    cap.release()
    out.release()

    # AUGMENTATION
    augmented_samples = []
    try:
        for f in sample_frames:
            augmented_samples += random_augmentations(f)
    except:
        augmented_samples = sample_frames

    df = pd.DataFrame(df_records).fillna(0)

    return df, output_path, augmented_samples, names_local
