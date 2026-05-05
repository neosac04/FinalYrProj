from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image input")

    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] < 3:
        raise ValueError(f"Unsupported channel count: {image.shape[2]}")

    # Default to the common OpenCV BGR -> RGB conversion.
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)


def _candidate_views(image: np.ndarray) -> list[np.ndarray]:
    if cv2 is None:
        return []

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return [rgb]

    if image.ndim != 3:
        return []

    base = image[:, :, :3]
    try:
        rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = base.copy()

    return [rgb, base.copy()]


def _log_face_boxes(detector_name: str, boxes: list[list[float]]) -> None:
    print(f"{detector_name} detected {len(boxes)} face(s)")
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        print(f"{detector_name} face {index}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


@lru_cache(maxsize=1)
def _get_face_detector():
    try:
        from retinaface import RetinaFace

        return ("retinaface", RetinaFace)
    except Exception:
        pass

    try:
        from facenet_pytorch import MTCNN

        return ("mtcnn", MTCNN(keep_all=True, device="cpu"))
    except Exception:
        return (None, None)


@lru_cache(maxsize=1)
def _get_haar_cascade():
    if cv2 is None:
        return None

    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    return cascade


def _pick_largest_bbox(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None

    def area(box: list[float]) -> float:
        x1, y1, x2, y2 = box[:4]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(boxes, key=area)


def _extract_boxes(detector_name: str, detector, rgb_image: np.ndarray) -> list[list[float]]:
    boxes: list[list[float]] = []

    if detector_name == "retinaface":
        faces = detector.detect_faces(rgb_image)
        if isinstance(faces, dict):
            for face in faces.values():
                region = face.get("facial_area") or face.get("box")
                if not region or len(region) < 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in region[:4]]
                boxes.append([x1, y1, x2, y2])
    else:
        detections = detector.detect(rgb_image)
        if detections is not None:
            bboxes = detections[0]
            if bboxes is not None:
                for box in bboxes:
                    if box is None or len(box) < 4:
                        continue
                    boxes.append([float(v) for v in box[:4]])

    _log_face_boxes(detector_name, boxes)
    return boxes


def _prepare_for_haar(rgb_image: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, float]:
    if cv2 is None:
        return rgb_image, 1.0

    height, width = rgb_image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return rgb_image, 1.0

    scale = max_side / float(largest_side)
    resized = cv2.resize(
        rgb_image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _extract_boxes_haar(rgb_image: np.ndarray) -> list[list[float]]:
    cascade = _get_haar_cascade()
    if cascade is None:
        return []

    resized_rgb, scale = _prepare_for_haar(rgb_image)
    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    boxes: list[list[float]] = []
    for x, y, w, h in faces:
        if scale != 1.0:
            x = int(round(x / scale))
            y = int(round(y / scale))
            w = int(round(w / scale))
            h = int(round(h / scale))
        boxes.append([float(x), float(y), float(x + w), float(y + h)])

    _log_face_boxes("haar", boxes)
    return boxes


def detect_largest_face(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Input:
        image: BGR or RGB image (numpy array)

    Output:
        Cropped face image (numpy array) OR None if no face found
    """
    try:
        rgb = _to_rgb(image)
    except Exception:
        return None

    if cv2 is None:
        print("OpenCV is unavailable; skipping face detection")
        return None

    detector_name, detector = _get_face_detector()
    use_haar = detector_name is None or detector is None

    try:
        boxes: list[list[float]] = []

        for candidate in _candidate_views(image):
            candidate_boxes = _extract_boxes(detector_name, detector, candidate) if not use_haar else _extract_boxes_haar(candidate)
            if candidate_boxes:
                boxes = candidate_boxes
                break

        if not boxes and not use_haar:
            for candidate in _candidate_views(image):
                candidate_boxes = _extract_boxes_haar(candidate)
                if candidate_boxes:
                    boxes = candidate_boxes
                    break

        largest = _pick_largest_bbox(boxes)
        if largest is None:
            print("Face detection returned no usable face box")
            return None

        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = largest
        x1 = max(0, int(np.floor(x1)))
        y1 = max(0, int(np.floor(y1)))
        x2 = min(w, int(np.ceil(x2)))
        y2 = min(h, int(np.ceil(y2)))

        if x2 <= x1 or y2 <= y1:
            print("Face detection returned an invalid bounding box")
            return None

        return rgb[y1:y2, x1:x2].copy()
    except Exception:
        return None


def test_face_detection(image_path: str):
    """
    Loads image, runs face detection, and displays:
    - original image
    - cropped face (if found)
    """
    import matplotlib.pyplot as plt

    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")

    image = cv2.imread(image_path)
    if image is None:
        print("No face detected")
        return None

    face = detect_largest_face(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if face is None:
        print("No face detected")
        blank = np.zeros((10, 10, 3), dtype=np.uint8)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(blank)
        plt.title("Cropped Face")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return None

    print("Face detected")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(face)
    plt.title("Cropped Face")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return face