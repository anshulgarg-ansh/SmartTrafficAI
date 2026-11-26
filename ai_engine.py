
import os
import gc
import cv2
import re
import uuid
import numpy as np

# Keep numerical libraries from creating many CPU threads on small instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from ultralytics import YOLO

try:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except Exception:
    torch = None

cv2.setNumThreads(1)


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
# MEMORY-AWARE MODEL INITIALIZATION
# ==========================================

# Models are loaded on first use, not when Flask imports this module.
# EasyOCR is intentionally loaded only when a valid plate crop is available.
vehicle_model = None
helmet_model = None
plate_model = None
ocr_reader = None


def get_vehicle_model():
    global vehicle_model
    if vehicle_model is None:
        print("Loading Vehicle Detection Model (CPU)...", flush=True)
        vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    return vehicle_model


def get_helmet_model():
    global helmet_model
    if helmet_model is None:
        print("Loading Helmet Detection Model (CPU)...", flush=True)
        helmet_model = YOLO(str(HELMET_MODEL_PATH))
    return helmet_model


def get_plate_model():
    global plate_model
    if plate_model is None:
        print("Loading Number Plate Detection Model (CPU)...", flush=True)
        plate_model = YOLO(str(PLATE_MODEL_PATH))
    return plate_model


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        print("Loading EasyOCR lazily (CPU)...", flush=True)
        import easyocr
        ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return ocr_reader


# Conservative limits for the Render Free 512 MB instance.
INFERENCE_IMAGE_SIZE = 416
SAMPLE_EVERY_N_FRAMES = 5
MAX_SAMPLED_FRAMES = 100
MAX_OCR_CALLS_PER_VIDEO = 12
MIN_PLATE_CONFIDENCE_FOR_OCR = 0.45


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
    """Run one lightweight OCR pass on a cropped plate image."""
    if plate_image is None or plate_image.size == 0:
        return "", 0.0

    try:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        # Avoid very large OCR tensors while keeping enough detail for plates.
        if width > 420:
            scale = 420 / max(width, 1)
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )
        elif width < 240:
            scale = 240 / max(width, 1)
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        processed = cv2.bilateralFilter(gray, 5, 35, 35)
        reader = get_ocr_reader()
        results = reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist=OCR_ALLOWLIST,
            decoder="greedy",
            batch_size=1,
            workers=0
        )

        if not results:
            return "", 0.0

        candidates = []
        ordered = sorted(
            results,
            key=lambda item: (
                (item[0][0][1] + item[0][1][1]) / 2,
                item[0][0][0]
            )
        )

        for detection in ordered:
            plate_text = normalize_indian_plate(detection[1])
            confidence = float(detection[2])
            if plate_text and confidence >= 0.20:
                candidates.append((plate_text, confidence))

        joined_text = "".join(item[1] for item in ordered)
        joined_confidence = sum(float(item[2]) for item in ordered) / max(len(ordered), 1)
        joined_plate = normalize_indian_plate(joined_text)
        if joined_plate and joined_confidence >= 0.20:
            candidates.append((joined_plate, joined_confidence))

        if not candidates:
            return "", 0.0

        scores = {}
        for plate_text, confidence in candidates:
            scores.setdefault(plate_text, []).append(confidence)

        best_text = max(
            scores,
            key=lambda plate: sum(scores[plate]) / len(scores[plate])
        )
        return best_text, sum(scores[best_text]) / len(scores[best_text])

    except Exception as e:
        print("OCR Error:", e, flush=True)
        return "", 0.0


# ==========================================
# PROCESS TRAFFIC VIDEO
# ==========================================

def process_video(video_path):

    global vehicle_model, helmet_model, plate_model, ocr_reader

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
    ocr_calls = 0

    evidence_images = []
    no_helmet_evidence = []
    plate_evidence = []
    plate_evidence_seen = set()

    # OCR observations are collected across sampled frames. Each plate gets
    # at most one vote per frame, preventing duplicate boxes in one frame
    # from artificially increasing its vote count.
    plate_votes = {}
    pending_ocr = []
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

            # Sample frames and cap total work to protect the small free instance.
            if frame_number % SAMPLE_EVERY_N_FRAMES != 0:
                continue

            if frames_processed >= MAX_SAMPLED_FRAMES:
                break

            frames_processed += 1

            annotated_frame = frame.copy()

            # ==================================
            # VEHICLE DETECTION
            # ==================================

            vehicle_results = get_vehicle_model().predict(
                frame,
                conf=0.35,
                imgsz=INFERENCE_IMAGE_SIZE,
                device="cpu",
                max_det=60,
                verbose=False
            )

            vehicle_result = vehicle_results[0]

            for box in vehicle_result.boxes:

                class_id = int(box.cls[0].item())

                if class_id in CLASS_NAMES:

                    class_name = CLASS_NAMES[class_id]

                    vehicle_counts[class_name] += 1

            annotated_frame = vehicle_result.plot()
            del vehicle_results, vehicle_result

            # ==================================
            # HELMET DETECTION
            # ==================================

            helmet_results = get_helmet_model().predict(
                frame,
                conf=0.35,
                imgsz=INFERENCE_IMAGE_SIZE,
                device="cpu",
                max_det=40,
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

            del helmet_results, helmet_result

            # ==================================
            # NUMBER PLATE DETECTION
            # ==================================

            plate_results = get_plate_model().predict(
                frame,
                conf=0.35,
                imgsz=INFERENCE_IMAGE_SIZE,
                device="cpu",
                max_det=30,
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

                # Queue a small number of strong plate crops. OCR is deferred
                # until YOLO inference is complete so OCR weights do not overlap
                # with all three detector models in RAM.
                if (
                    plate_confidence >= MIN_PLATE_CONFIDENCE_FOR_OCR
                    and len(pending_ocr) < MAX_OCR_CALLS_PER_VIDEO
                ):
                    pending_ocr.append({
                        "crop": plate_crop.copy(),
                        "plate_confidence": plate_confidence,
                        "frame_number": frame_number
                    })

                plate_detections += 1
                label = "Plate Candidate"

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

            del plate_results, plate_result

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

            # Release large per-frame arrays before reading the next frame.
            del frame, annotated_frame
            if frames_processed % 10 == 0:
                gc.collect()

    finally:

        cap.release()
        gc.collect()

    # ==========================================
    # VALIDATE VIDEO
    # ==========================================

    if frames_processed == 0:

        raise ValueError(
            "Video is empty or too short to process."
        )

    # ==========================================
    # RELEASE DETECTORS BEFORE LOADING OCR
    # ==========================================

    # Free YOLO model weights before EasyOCR initializes its own Torch models.
    # This reduces peak memory on Render's 512 MB free instance.
    vehicle_model = None
    helmet_model = None
    plate_model = None
    gc.collect()

    # Run OCR only after detector memory has been released.
    ocr_calls = 0
    events_by_frame = {event["frame_number"]: event for event in no_helmet_events}

    for pending in pending_ocr:
        crop = pending["crop"]
        try:
            ocr_calls += 1
            plate_text, ocr_confidence = read_number_plate(crop)
            if not plate_text:
                continue

            frame_no = pending["frame_number"]
            plate_confidence = pending["plate_confidence"]
            plate_record = {
                "vehicle_number": plate_text,
                "plate_confidence": plate_confidence,
                "ocr_confidence": ocr_confidence,
                "frame_number": frame_no
            }

            # One vote per exact plate string per sampled frame.
            frame_votes = plate_votes.setdefault(plate_text, [])
            if not any(item["frame_number"] == frame_no for item in frame_votes):
                frame_votes.append({
                    "frame_number": frame_no,
                    "plate_confidence": plate_confidence,
                    "ocr_confidence": ocr_confidence
                })

            # Attach only as a same-frame review candidate, not a confirmed match.
            event = events_by_frame.get(frame_no)
            if event is not None:
                event["plate_candidates"].append(plate_record)

            # Save at most 12 validated plate crops for the evidence gallery.
            if plate_text not in plate_evidence_seen and len(plate_evidence) < 12:
                filename = f"plate_{run_id}_{len(plate_evidence) + 1}.jpg"
                save_path = EVIDENCE_DIR / filename
                if cv2.imwrite(str(save_path), crop):
                    plate_evidence_seen.add(plate_text)
                    plate_evidence.append({
                        "image": f"evidence/{filename}",
                        "vehicle_number": plate_text,
                        "plate_confidence": plate_confidence,
                        "ocr_confidence": ocr_confidence
                    })
        finally:
            del crop

    pending_ocr.clear()

    # Release OCR weights after this video so the web worker can return to a
    # lower idle-memory state between uploads.
    ocr_reader = None
    gc.collect()

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

        "ocr_calls": ocr_calls,

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
