
import cv2
import re
import uuid
import easyocr
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

EVIDENCE_DIR = BASE_DIR / "static" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_MODEL_PATH = BASE_DIR / "yolov8n.pt"

HELMET_MODEL_PATH = BASE_DIR / "models" / "helmet_model.pt"

PLATE_MODEL_PATH = BASE_DIR / "models" / "license_plate.pt"


# ==========================================
# MODEL FILE VALIDATION
# ==========================================

for model_path in (
    VEHICLE_MODEL_PATH,
    HELMET_MODEL_PATH,
    PLATE_MODEL_PATH
):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required AI model not found: {model_path}"
        )
    if model_path.stat().st_size == 0:
        raise ValueError(
            f"AI model file is empty or corrupted: {model_path}"
        )


# ==========================================
# LOAD AI MODELS
# ==========================================

print("Loading Vehicle Detection Model...")
vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))

print("Loading Helmet Detection Model...")
helmet_model = YOLO(str(HELMET_MODEL_PATH))

print("Loading Number Plate Detection Model...")
plate_model = YOLO(str(PLATE_MODEL_PATH))

print("Loading EasyOCR...")

ocr_reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("ALL AI MODELS LOADED SUCCESSFULLY!")


# ==========================================
# VEHICLE CLASSES
# ==========================================

CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ==========================================
# HELMET CLASSES
# ==========================================

HELMET_CLASSES = {
    0: "With Helmet",
    1: "Without Helmet"
}


# ==========================================
# INDIAN NUMBER PLATE VALIDATION + OCR
# ==========================================

INDIAN_PLATE_PATTERN = re.compile(
    r"^(?:[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$"
)

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"



def edit_distance(a, b):
    """Return Levenshtein edit distance for two short registration strings."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b)
            ))
        previous = current
    return previous[-1]


def observation_quality(observations):
    """A ranking score for OCR review; not a legal-confidence probability."""
    if not observations:
        return 0.0
    ocr = sum(x["ocr_confidence"] for x in observations) / len(observations)
    plate = sum(x["plate_confidence"] for x in observations) / len(observations)
    return 0.60 * ocr + 0.40 * plate


def build_plate_consensus(plate_votes, max_frame_gap=25):
    """Conservatively merge likely one-off OCR typos into repeated readings.

    A singleton is merged only when a stronger candidate has at least two
    frame votes, the strings are very close, the candidate is lower quality,
    and their observations occur near each other in the video. Repeated
    competing strings are kept separate to reduce false vehicle merges.
    """
    records = []
    for text, observations in plate_votes.items():
        records.append({
            "vehicle_number": text,
            "observations": list(observations),
            "vote_count": len(observations),
            "quality": observation_quality(observations),
            "variants": set()
        })

    # Strong/repeated readings become anchors before single-frame readings.
    records.sort(
        key=lambda x: (x["vote_count"], x["quality"]),
        reverse=True
    )

    anchors = []
    for record in records:
        best_anchor = None
        best_distance = None

        if record["vote_count"] == 1:
            candidate_frame = record["observations"][0]["frame_number"]
            for anchor in anchors:
                if anchor["vote_count"] < 2:
                    continue
                # Allow a single OCR insertion/deletion as well as a
                # substitution (for example, one duplicated character).
                if abs(len(record["vehicle_number"]) - len(anchor["vehicle_number"])) > 1:
                    continue

                distance = edit_distance(
                    record["vehicle_number"], anchor["vehicle_number"]
                )
                # Permit only one edit; do not merge broad similarities.
                if distance > 1:
                    continue

                anchor_frames = [
                    item["frame_number"] for item in anchor["observations"]
                ]
                frame_gap = min(abs(candidate_frame - f) for f in anchor_frames)
                if frame_gap > max_frame_gap:
                    continue

                # A one-off candidate must be noticeably weaker than the
                # repeated anchor before it can be treated as an OCR variant.
                if record["quality"] > anchor["quality"] - 0.04:
                    continue

                if best_anchor is None or distance < best_distance:
                    best_anchor = anchor
                    best_distance = distance

        if best_anchor is not None:
            best_anchor["observations"].extend(record["observations"])
            best_anchor["vote_count"] = len(best_anchor["observations"])
            best_anchor["variants"].add(record["vehicle_number"])
            best_anchor["variants"].update(record["variants"])
        else:
            anchors.append(record)

    consensus = []
    raw_to_canonical = {}

    for anchor in anchors:
        observations = sorted(
            anchor["observations"], key=lambda x: x["frame_number"]
        )
        canonical = anchor["vehicle_number"]
        raw_to_canonical[canonical] = canonical
        for variant in anchor["variants"]:
            raw_to_canonical[variant] = canonical

        vote_count = len(observations)
        avg_ocr = sum(x["ocr_confidence"] for x in observations) / vote_count
        avg_plate = sum(x["plate_confidence"] for x in observations) / vote_count

        consensus.append({
            "vehicle_number": canonical,
            "vote_count": vote_count,
            "plate_confidence": avg_plate,
            "ocr_confidence": avg_ocr,
            "combined_confidence": 0.60 * avg_ocr + 0.40 * avg_plate,
            "frame_number": observations[-1]["frame_number"],
            "frames_seen": [x["frame_number"] for x in observations],
            "ocr_variants": sorted(anchor["variants"])
        })

    consensus.sort(
        key=lambda x: (x["vote_count"], x["combined_confidence"]),
        reverse=True
    )
    return consensus, raw_to_canonical

def normalize_indian_plate(raw_text):
    """
    Normalize OCR output and accept only common Indian registration formats.

    Standard: XX 00 XX 0000
    BH series: 00 BH 0000 XX

    Common OCR character confusions are corrected according to each
    registration-number position. Invalid text is rejected.
    """
    text = re.sub(r"[^A-Z0-9]", "", str(raw_text).upper())

    if not text:
        return ""

    # BH-series registration: 2 digits + BH + 4 digits + 1-2 letters.
    if (
        len(text) in (9, 10)
        and text[2:4].translate(str.maketrans({"8": "B"})) == "BH"
    ):
        bh_candidate = (
            text[:2].translate(str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1"}))
            + "BH"
            + text[4:8].translate(str.maketrans({
                "O": "0", "Q": "0", "D": "0",
                "I": "1", "L": "1",
                "Z": "2", "S": "5", "B": "8", "G": "6"
            }))
            + text[8:].translate(str.maketrans({"0": "O", "1": "I"}))
        )
        if INDIAN_PLATE_PATTERN.fullmatch(bh_candidate):
            return bh_candidate

    # Standard Indian registration: state + district + series + 4 digits.
    # Try the possible district/series lengths and correct OCR confusions
    # only in positions that are expected to contain letters or digits.
    for district_len in (2, 1):
        for series_len in (3, 2, 1):
            expected_len = 2 + district_len + series_len + 4
            if len(text) != expected_len:
                continue

            state = text[:2].translate(str.maketrans({
                "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"
            }))
            district = text[2:2 + district_len].translate(str.maketrans({
                "O": "0", "Q": "0", "D": "0",
                "I": "1", "L": "1",
                "Z": "2", "S": "5", "B": "8", "G": "6"
            }))
            series_start = 2 + district_len
            series = text[series_start:series_start + series_len].translate(
                str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"})
            )
            final_digits = text[-4:].translate(str.maketrans({
                "O": "0", "Q": "0", "D": "0",
                "I": "1", "L": "1",
                "Z": "2", "S": "5", "B": "8", "G": "6"
            }))

            candidate = state + district + series + final_digits
            if INDIAN_PLATE_PATTERN.fullmatch(candidate):
                return candidate

    return ""


def read_number_plate(plate_image):
    """
    Read a plate using multiple preprocessing variants.
    Returns only a format-valid Indian registration number.
    """
    if plate_image is None or plate_image.size == 0:
        return "", 0.0

    try:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Upscale small plate crops to give OCR more character detail.
        height, width = gray.shape[:2]
        if width < 600:
            scale = 600 / max(width, 1)
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # Improve local contrast and reduce noise.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(gray)

        denoised = cv2.bilateralFilter(clahe, 9, 75, 75)

        sharpen_kernel = np.array(
            [
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ],
            dtype=np.float32
        )
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

        _, otsu = cv2.threshold(
            sharpened, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        variants = [denoised, sharpened, otsu, adaptive]
        valid_candidates = []

        for processed in variants:
            results = ocr_reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist=OCR_ALLOWLIST,
                decoder="greedy"
            )

            if not results:
                continue

            # Try each OCR segment separately.
            for detection in results:
                raw = detection[1]
                confidence = float(detection[2])
                plate_text = normalize_indian_plate(raw)

                if plate_text and confidence >= 0.20:
                    valid_candidates.append((plate_text, confidence))

            # Also try joining text segments in reading order. Some plate
            # crops are returned as multiple OCR boxes.
            ordered = sorted(
                results,
                key=lambda item: (
                    (item[0][0][1] + item[0][1][1]) / 2,
                    item[0][0][0]
                )
            )
            joined_text = "".join(item[1] for item in ordered)
            joined_confidence = sum(
                float(item[2]) for item in ordered
            ) / max(len(ordered), 1)

            joined_plate = normalize_indian_plate(joined_text)
            if joined_plate and joined_confidence >= 0.20:
                valid_candidates.append(
                    (joined_plate, joined_confidence)
                )

        if not valid_candidates:
            return "", 0.0

        # Prefer the strongest OCR result; repeated agreement is a useful
        # tie-breaker when multiple preprocessing variants produce a result.
        scores = {}
        for plate_text, confidence in valid_candidates:
            scores.setdefault(plate_text, []).append(confidence)

        best_text = max(
            scores,
            key=lambda plate: (
                sum(scores[plate]) / len(scores[plate])
                + 0.03 * (len(scores[plate]) - 1)
            )
        )
        best_confidence = sum(scores[best_text]) / len(scores[best_text])

        return best_text, best_confidence

    except Exception as e:
        print("OCR Error:", e)
        return "", 0.0


# ==========================================
# PROCESS TRAFFIC VIDEO
# ==========================================

def process_video(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise ValueError(
            "Unable to open uploaded video."
        )

    vehicle_counts = {
        "Person": 0,
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    helmet_counts = {
        "With Helmet": 0,
        "Without Helmet": 0
    }

    frames_processed = 0
    frame_number = 0

    plate_detections = 0

    evidence_images = []
    no_helmet_evidence = []
    plate_evidence = []
    plate_evidence_seen = set()

    # OCR observations are collected across sampled frames. Each plate gets
    # at most one vote per frame, preventing duplicate boxes in one frame
    # from artificially increasing its vote count.
    plate_votes = {}
    recognized_plates = []

    no_helmet_events = []

    # Unique filename prefix for each processing run
    run_id = uuid.uuid4().hex[:8]

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Process every fifth frame
            if frame_number % 5 != 0:
                continue

            frames_processed += 1

            annotated_frame = frame.copy()

            # ==================================
            # VEHICLE DETECTION
            # ==================================

            vehicle_results = vehicle_model.predict(
                frame,
                conf=0.35,
                imgsz=640,
                verbose=False
            )

            vehicle_result = vehicle_results[0]

            for box in vehicle_result.boxes:

                class_id = int(box.cls[0].item())

                if class_id in CLASS_NAMES:

                    class_name = CLASS_NAMES[class_id]

                    vehicle_counts[class_name] += 1

            annotated_frame = vehicle_result.plot()

            # ==================================
            # HELMET DETECTION
            # ==================================

            helmet_results = helmet_model.predict(
                frame,
                conf=0.35,
                imgsz=640,
                verbose=False
            )

            helmet_result = helmet_results[0]

            frame_has_no_helmet = False
            frame_has_helmet_detection = False

            frame_no_helmet_records = []

            for box in helmet_result.boxes:

                class_id = int(box.cls[0].item())

                if class_id not in HELMET_CLASSES:
                    continue

                class_name = HELMET_CLASSES[class_id]

                confidence = float(box.conf[0].item())

                helmet_counts[class_name] += 1

                frame_has_helmet_detection = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                if class_id == 1:

                    frame_has_no_helmet = True

                    label_color = (0, 0, 255)

                    frame_no_helmet_records.append({
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })

                else:

                    label_color = (0, 255, 0)

                label = (
                    f"{class_name}: {confidence:.2f}"
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    label_color,
                    2
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    label_color,
                    2
                )

            # ==================================
            # NUMBER PLATE DETECTION
            # ==================================

            plate_results = plate_model.predict(
                frame,
                conf=0.35,
                imgsz=640,
                verbose=False
            )

            plate_result = plate_results[0]

            frame_plate_candidates = []
            voted_this_frame = set()

            for box in plate_result.boxes:

                plate_confidence = float(
                    box.conf[0].item()
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Keep coordinates inside frame
                h, w = frame.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                plate_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                # Extract and validate the registration number.
                plate_text, ocr_confidence = read_number_plate(plate_crop)

                plate_detections += 1

                # Never display arbitrary OCR text as a registration number.
                label = plate_text if plate_text else "Unverified Plate"

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 165, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 165, 0),
                    2
                )

                # Keep only format-valid OCR results as review candidates.
                if plate_text:
                    plate_record = {
                        "vehicle_number": plate_text,
                        "plate_confidence": plate_confidence,
                        "ocr_confidence": ocr_confidence,
                        "frame_number": frame_number
                    }

                    frame_plate_candidates.append(plate_record)

                    # Save only validated plate crops; this prevents words
                    # such as CAMERA or GOODS CARRIER appearing as plate evidence.
                    if (
                        plate_text not in plate_evidence_seen
                        and len(plate_evidence) < 12
                    ):
                        filename = (
                            f"plate_{run_id}_"
                            f"{len(plate_evidence) + 1}.jpg"
                        )
                        save_path = EVIDENCE_DIR / filename

                        if cv2.imwrite(str(save_path), plate_crop):
                            plate_evidence_seen.add(plate_text)
                            plate_evidence.append({
                                "image": f"evidence/{filename}",
                                "vehicle_number": plate_text,
                                "plate_confidence": plate_confidence,
                                "ocr_confidence": ocr_confidence
                            })

                    # Multi-frame voting: one vote per distinct plate per frame.
                    if plate_text not in voted_this_frame:
                        voted_this_frame.add(plate_text)
                        plate_votes.setdefault(plate_text, []).append({
                            "frame_number": frame_number,
                            "plate_confidence": plate_confidence,
                            "ocr_confidence": ocr_confidence
                        })

            # ==================================
            # SAVE GENERAL EVIDENCE
            # ==================================

            if (
                frame_has_helmet_detection
                and len(evidence_images) < 6
            ):

                filename = (
                    f"evidence_{run_id}_"
                    f"{len(evidence_images) + 1}.jpg"
                )

                save_path = EVIDENCE_DIR / filename

                if cv2.imwrite(
                    str(save_path),
                    annotated_frame
                ):

                    evidence_images.append(
                        f"evidence/{filename}"
                    )

            # ==================================
            # SAVE NO HELMET EVIDENCE
            # ==================================

            if (
                frame_has_no_helmet
                and len(no_helmet_events) < 30
            ):

                filename = (
                    f"no_helmet_{run_id}_"
                    f"{len(no_helmet_evidence) + 1}.jpg"
                )

                save_path = EVIDENCE_DIR / filename

                if cv2.imwrite(
                    str(save_path),
                    annotated_frame
                ):

                    no_helmet_evidence.append(
                        f"evidence/{filename}"
                    )

                    # Keep same-frame plate candidates
                    # for manual review, not automatic matching
                    no_helmet_events.append({

                        "frame_number": frame_number,

                        "evidence_path": (
                            f"evidence/{filename}"
                        ),

                        "helmet_confidence": max(
                            (
                                record["confidence"]
                                for record in frame_no_helmet_records
                            ),
                            default=0.0
                        ),

                        "plate_candidates": (
                            frame_plate_candidates
                        ),

                        "review_status": "Pending Review"

                    })

    finally:

        cap.release()

    # ==========================================
    # VALIDATE VIDEO
    # ==========================================

    if frames_processed == 0:

        raise ValueError(
            "Video is empty or too short to process."
        )

    # ==========================================
    # FINAL OCR CONSENSUS + DUPLICATE REDUCTION
    # ==========================================

    recognized_plates, raw_to_canonical = build_plate_consensus(
        plate_votes,
        max_frame_gap=25
    )

    # Re-map same-frame review candidates to the consensus spelling while
    # preserving separate candidates when no safe merge was justified.
    for event in no_helmet_events:
        unique_candidates = {}
        for candidate in event["plate_candidates"]:
            raw_number = candidate["vehicle_number"]
            canonical = raw_to_canonical.get(raw_number, raw_number)
            candidate = dict(candidate)
            candidate["vehicle_number"] = canonical
            current = unique_candidates.get(canonical)
            if current is None or (
                candidate["ocr_confidence"] + candidate["plate_confidence"]
                > current["ocr_confidence"] + current["plate_confidence"]
            ):
                unique_candidates[canonical] = candidate
        event["plate_candidates"] = list(unique_candidates.values())

    # Keep one strongest evidence crop per final consensus registration.
    # Evidence files already written to disk are harmless if not referenced.
    best_evidence = {}
    for item in plate_evidence:
        canonical = raw_to_canonical.get(
            item["vehicle_number"], item["vehicle_number"]
        )
        candidate_score = (
            0.60 * item["ocr_confidence"]
            + 0.40 * item["plate_confidence"]
        )
        existing = best_evidence.get(canonical)
        if existing is None or candidate_score > existing["score"]:
            best_evidence[canonical] = {
                "score": candidate_score,
                "item": {
                    **item,
                    "vehicle_number": canonical
                }
            }
    plate_evidence = [entry["item"] for entry in best_evidence.values()]

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    total_vehicle_detections = sum(
        vehicle_counts.values()
    )

    total_helmet_detections = sum(
        helmet_counts.values()
    )

    return {

        # Vehicle results
        "counts": vehicle_counts,

        "frames_processed": frames_processed,

        "total_detections": total_vehicle_detections,

        # Helmet results
        "helmet_counts": helmet_counts,

        "total_helmet_detections": total_helmet_detections,

        "no_helmet_detections": (
            helmet_counts["Without Helmet"]
        ),

        # Vehicle evidence
        "evidence_images": evidence_images,

        # Helmet evidence
        "no_helmet_evidence": no_helmet_evidence,

        # ANPR results
        "plate_detections": plate_detections,

        "recognized_plates": recognized_plates,

        "plate_evidence": plate_evidence,

        # Review candidates
        "no_helmet_events": no_helmet_events

    }
