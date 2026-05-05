from __future__ import annotations
import os
import numpy as np
from PIL import Image
from app.schemas.response import FacialAnalysis

try:
    import cv2
except Exception:
    cv2 = None

try:
    from scipy.spatial import distance as scipy_dist
except Exception:
    class _DistanceFallback:
        @staticmethod
        def euclidean(point_a, point_b):
            delta = np.asarray(point_a, dtype=np.float64) - np.asarray(point_b, dtype=np.float64)
            return float(np.sqrt(np.sum(delta * delta)))

    scipy_dist = _DistanceFallback()

# Default path for the MediaPipe FaceLandmarker model bundle
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../models/mediapipe/face_landmarker.task",
)


class FacialAnalyzer:
    """
    MediaPipe FaceLandmarker (Tasks API, mediapipe >=0.10).
    Computes anatomical forgery signals:
    - Eye specular highlight symmetry
    - Iris circularity (5-point iris landmarks 468-477)
    - Jaw boundary gradient discontinuity (face swap seam)
    - Facial geometry golden-ratio consistency
    - Landmark left/right symmetry score
    """

    _instance: FacialAnalyzer | None = None

    def __init__(self, model_path: str | None = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        path = model_path or _DEFAULT_MODEL_PATH
        path = os.path.abspath(path)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker model not found at {path}. "
                "Run: python models/download_weights.py"
            )

        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    @classmethod
    def get_instance(cls, model_path: str | None = None) -> FacialAnalyzer:
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def analyze(self, image: Image.Image) -> FacialAnalysis:
        img_rgb = np.array(image.convert("RGB"))
        h, w = img_rgb.shape[:2]

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=img_rgb,
        )
        detection = self._landmarker.detect(mp_image)

        if not detection.face_landmarks:
            return FacialAnalysis(
                face_detected=False,
                face_count=0,
                landmark_consistency_score=0.5,
                eye_reflection_symmetry=0.5,
                iris_regularity_score=0.5,
                facial_geometry_score=0.5,
                blending_boundary_score=0.5,
                landmark_points=[],
            )

        face_count = len(detection.face_landmarks)
        # Analyse the most prominent (first) face
        raw_lm = detection.face_landmarks[0]
        pts = [(lm.x * w, lm.y * h) for lm in raw_lm]

        return FacialAnalysis(
            face_detected=True,
            face_count=face_count,
            landmark_consistency_score=self._landmark_symmetry(pts),
            eye_reflection_symmetry=self._eye_reflection_symmetry(img_rgb, pts),
            iris_regularity_score=self._iris_regularity(pts),
            facial_geometry_score=self._golden_ratio_score(pts),
            blending_boundary_score=self._jaw_blending_score(img_rgb, pts),
            landmark_points=[[p[0], p[1]] for p in pts[:68]],
        )

    # ------------------------------------------------------------------
    def _landmark_symmetry(self, pts: list[tuple]) -> float:
        nose_tip = pts[1]
        midline_x = nose_tip[0]
        pairs = [(33, 263), (130, 359), (234, 454), (172, 397), (61, 291)]
        diffs = []
        for l_idx, r_idx in pairs:
            if l_idx >= len(pts) or r_idx >= len(pts):
                continue
            l_dist = abs(pts[l_idx][0] - midline_x)
            r_dist = abs(pts[r_idx][0] - midline_x)
            if l_dist + r_dist > 1e-6:
                diffs.append(abs(l_dist - r_dist) / (l_dist + r_dist))
        if not diffs:
            return 0.5
        return float(max(0.0, 1.0 - np.mean(diffs) * 2))

    def _eye_reflection_symmetry(self, img: np.ndarray, pts: list[tuple]) -> float:
        try:
            def highlight_centroid(indices: list[int]):
                eye_pts = np.array([pts[i] for i in indices if i < len(pts)], dtype=np.int32)
                if len(eye_pts) == 0:
                    return None
                x1, y1 = eye_pts.min(axis=0) - 4
                x2, y2 = eye_pts.max(axis=0) + 4
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    return None
                ys, xs = np.where(roi.mean(axis=2) > 220)
                if len(ys) == 0:
                    return None
                return (xs.mean() / max(x2 - x1, 1), ys.mean() / max(y2 - y1, 1))

            left_lm  = list(range(33, 42)) + list(range(159, 163)) + [133]
            right_lm = list(range(263, 272)) + list(range(386, 390)) + [362]
            lc = highlight_centroid(left_lm)
            rc = highlight_centroid(right_lm)
            if lc is None or rc is None:
                return 0.5
            mirrored_rx = 1.0 - rc[0]
            x_diff = abs(lc[0] - mirrored_rx)
            y_diff = abs(lc[1] - rc[1])
            return float(max(0.0, 1.0 - (x_diff + y_diff) / 2 * 3))
        except Exception:
            return 0.5

    def _iris_regularity(self, pts: list[tuple]) -> float:
        if len(pts) < 478:
            return 0.5

        def circularity(indices: list[int]) -> float:
            iris_pts = np.array([pts[i] for i in indices])
            center = iris_pts.mean(axis=0)
            radii = [scipy_dist.euclidean(p, center) for p in iris_pts]
            mean_r = np.mean(radii)
            if mean_r < 1e-6:
                return 0.5
            cv = np.std(radii) / mean_r
            return float(max(0.0, 1.0 - cv * 5))

        left  = circularity(list(range(468, 473)))
        right = circularity(list(range(473, 478)))
        return (left + right) / 2

    def _golden_ratio_score(self, pts: list[tuple]) -> float:
        try:
            phi = 1.618
            face_h = scipy_dist.euclidean(pts[10], pts[152])
            face_w = scipy_dist.euclidean(pts[234], pts[454])
            if face_w < 1e-6:
                return 0.5
            deviation = abs(face_h / face_w - phi) / phi
            return float(max(0.0, 1.0 - deviation))
        except Exception:
            return 0.5

    def _jaw_blending_score(self, img: np.ndarray, pts: list[tuple]) -> float:
        try:
            if cv2 is None:
                return 0.5

            jaw_indices = [10, 338, 297, 332, 284, 251, 389, 356,
                           454, 323, 361, 288, 397, 365, 379, 378,
                           400, 377, 152]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            magnitudes = []
            for idx in jaw_indices:
                if idx >= len(pts):
                    continue
                x, y = int(pts[idx][0]), int(pts[idx][1])
                if 0 <= y < laplacian.shape[0] and 0 <= x < laplacian.shape[1]:
                    magnitudes.append(abs(laplacian[y, x]))
            if not magnitudes:
                return 0.5
            mean_mag = np.mean(magnitudes)
            return float(np.clip(1.0 - (mean_mag - 10) / 80, 0.0, 1.0))
        except Exception:
            return 0.5
