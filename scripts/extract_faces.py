from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "images" / "raw"
PROCESSED_DIR = BASE_DIR / "images" / "processed"
TARGET_SIZE = (256, 256)
FACE_MARGIN_RATIO = 0.35
FACE_VERTICAL_OFFSET_RATIO = 0.12
DETECTION_MAX_DIM = 1200


class FaceDetector:
    def __init__(self) -> None:
        cascade_dir = Path(cv2.data.haarcascades)
        self.cv_detectors = [
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml")),
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_alt.xml")),
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_alt2.xml")),
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml")),
        ]

    def _score_face(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_width: int,
        image_height: int,
    ) -> float:
        area_score = float(w * h)
        face_center_x = x + (w / 2)
        face_center_y = y + (h / 2)
        image_center_x = image_width / 2
        image_center_y = image_height / 2

        normalized_dx = abs(face_center_x - image_center_x) / max(image_width, 1)
        normalized_dy = abs(face_center_y - image_center_y) / max(image_height, 1)
        center_penalty = (normalized_dx * 0.6) + (normalized_dy * 0.4)

        return area_score * (1.0 - min(center_penalty, 0.75))

    def detect(self, rgb_pixels: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        faces: list[tuple[int, int, int, int, float]] = []
        image_height, image_width = rgb_pixels.shape[:2]
        scale = 1.0
        detection_pixels = rgb_pixels

        largest_dim = max(image_width, image_height)
        if largest_dim > DETECTION_MAX_DIM:
            scale = DETECTION_MAX_DIM / largest_dim
            detection_pixels = cv2.resize(
                rgb_pixels,
                (int(image_width * scale), int(image_height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        detection_height, detection_width = detection_pixels.shape[:2]

        gray = cv2.cvtColor(detection_pixels, cv2.COLOR_RGB2GRAY)
        enhanced_variants = [
            gray,
            cv2.equalizeHist(gray),
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
        ]
        seen_boxes: set[tuple[int, int, int, int]] = set()

        for variant in enhanced_variants:
            for detector_index, detector in enumerate(self.cv_detectors):
                detections = detector.detectMultiScale(
                    variant,
                    scaleFactor=1.05,
                    minNeighbors=4 if detector_index < 3 else 3,
                    minSize=(max(24, int(36 * scale)), max(24, int(36 * scale))),
                )

                for (x, y, w, h) in detections:
                    box = (
                        int(round(x / scale)),
                        int(round(y / scale)),
                        int(round(w / scale)),
                        int(round(h / scale)),
                    )
                    if box in seen_boxes:
                        continue
                    seen_boxes.add(box)
                    faces.append(
                        (
                            box[0],
                            box[1],
                            box[2],
                            box[3],
                            self._score_face(*box, image_width, image_height),
                        )
                    )

                if detector_index == 3:
                    flipped = cv2.flip(variant, 1)
                    profile_detections = detector.detectMultiScale(
                        flipped,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(max(24, int(36 * scale)), max(24, int(36 * scale))),
                    )
                    for (x, y, w, h) in profile_detections:
                        translated_x = detection_width - x - w
                        box = (
                            int(round(translated_x / scale)),
                            int(round(y / scale)),
                            int(round(w / scale)),
                            int(round(h / scale)),
                        )
                        if box in seen_boxes:
                            continue
                        seen_boxes.add(box)
                        faces.append(
                            (
                                box[0],
                                box[1],
                                box[2],
                                box[3],
                                self._score_face(*box, image_width, image_height),
                            )
                        )

        return faces


detector = FaceDetector()
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def extract_face(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image)
    image_height, image_width = pixels.shape[:2]

    results = detector.detect(pixels)
    if not results:
        return None

    results = sorted(results, key=lambda face: face[4], reverse=True)
    x, y, w, h, _score = results[0]

    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    margin = int(max(w, h) * FACE_MARGIN_RATIO)
    crop_size = max(w, h) + (2 * margin)
    center_x = x + (w / 2)
    center_y = y + (h / 2) - (h * FACE_VERTICAL_OFFSET_RATIO)

    left = int(round(center_x - (crop_size / 2)))
    top = int(round(center_y - (crop_size / 2)))
    right = left + crop_size
    bottom = top + crop_size

    if left < 0:
        right += -left
        left = 0
    if top < 0:
        bottom += -top
        top = 0
    if right > image_width:
        left = max(0, left - (right - image_width))
        right = image_width
    if bottom > image_height:
        top = max(0, top - (bottom - image_height))
        bottom = image_height

    face = pixels[top:bottom, left:right]
    if face.size == 0:
        return None

    return Image.fromarray(face).resize(TARGET_SIZE, Image.Resampling.LANCZOS)


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {RAW_DIR}")

    for person_dir in RAW_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        output_dir = PROCESSED_DIR / person_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        skipped = 0

        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            try:
                face = extract_face(img_path)
                if face is None:
                    skipped += 1
                    continue

                out_path = output_dir / f"{img_path.stem}_face.jpg"
                face.save(out_path, quality=95)
                count += 1
            except Exception as e:
                print(f"Erreur sur {img_path.name}: {e}")
                skipped += 1

        print(f"{person_dir.name}: {count} visages extraits, {skipped} ignores")


if __name__ == "__main__":
    main()
