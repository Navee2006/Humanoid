"""
orin1/face_detector.py
────────────────────────────────────────────────────────────────
VEDHA – Face Recognition Detector

Adapted from the provided face_recognition ROS node, decoupled
from ROS so it works with raw OpenCV frames from any source
(direct camera capture OR ROS /camera/color/image_raw topic).

Responsibilities:
  • Load registered face encodings from config on startup
  • On each frame: detect + compare all visible faces
  • Return structured results: list of FaceDetection(name, bbox, distance)
  • Maintain per-face cooldown and consecutive-frame counters
  • Emit KNOWN_FACE_DETECTED or UNKNOWN_PERSON_DETECTED via callback

The caller (orin1/main.py) feeds frames here and acts on results.
"""

from __future__ import annotations
import logging, os, time, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import face_recognition
import numpy as np
import yaml

log = logging.getLogger("orin1.face")

with open(Path(__file__).parent.parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

FRCFG = CFG["perception"]["face_recognition"]


# ═══════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════
@dataclass
class FaceDetection:
    """Single detected face in a frame."""
    name: str               # registered name OR "Unknown"
    title: str              # optional title from config (e.g. "Chairman")
    distance: float         # face_recognition distance (lower = more confident)
    bbox: tuple             # (top, right, bottom, left) in ORIGINAL frame coords
    is_known: bool          # True if matched to a registered face
    session_id: str         # stable per known person; random for unknowns


@dataclass
class _FaceRecord:
    """Runtime state tracked per registered face."""
    name: str
    title: str
    encoding: np.ndarray
    last_trigger: float = 0.0
    consec_frames: int  = 0


@dataclass
class _UnknownRecord:
    """Runtime state for unknown-person cooldown."""
    last_trigger: float = 0.0
    consec_frames: int  = 0
    session_id: str     = field(default_factory=lambda: f"unk_{uuid.uuid4().hex[:6]}")


# ═══════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════
class FaceRecognitionDetector:
    """
    Drop-in replacement for the old YOLO PersonDetector.
    Call process_frame(frame) with each BGR numpy array.
    Register a trigger callback via set_trigger_callback().
    """

    def __init__(self):
        self._known: list[_FaceRecord] = []
        self._unknown = _UnknownRecord()

        self._scale      = FRCFG["scale"]              # 0.25
        self._tolerance  = FRCFG["tolerance"]          # 0.55
        self._cd_known   = FRCFG["cooldown_per_face"]  # 30s
        self._cd_unknown = FRCFG["cooldown_unknown"]   # 20s
        self._min_frames = FRCFG["min_consecutive_frames"]  # 3

        self._trigger_cb: Optional[Callable] = None    # set by main.py

        self._load_known_faces()

    # ── Setup ───────────────────────────────────────────────────
    def _load_known_faces(self):
        entries = FRCFG.get("known_faces", [])
        for entry in entries:
            name  = entry["name"]
            path  = entry["image_path"]
            title = entry.get("title", "")

            if not os.path.exists(path):
                log.error(f"Face image NOT FOUND: {path} (skipping {name})")
                continue

            log.info(f"Loading face: {name} from {path}")
            img       = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)

            if len(encodings) == 0:
                log.error(f"No face found in {path} – check the image (clear frontal photo needed)")
                continue

            self._known.append(_FaceRecord(
                name=name, title=title, encoding=encodings[0]
            ))

        log.info(f"Loaded {len(self._known)} registered faces: "
                 f"{[r.name for r in self._known]}")

    def set_trigger_callback(self, cb: Callable):
        """
        Callback signature:
            cb(detection: FaceDetection, frame: np.ndarray)
        Called when a face clears the consecutive-frame threshold
        AND the per-face cooldown has elapsed.
        """
        self._trigger_cb = cb

    # ── Main processing entry point ─────────────────────────────
    def process_frame(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, list[FaceDetection]]:
        """
        Run face detection + recognition on a BGR frame.
        Returns (annotated_frame, list[FaceDetection]).
        Fires trigger callback for qualifying detections.
        """
        # ── Resize for speed (matches original 0.25 scale) ─────
        small = cv2.resize(bgr_frame, (0, 0), fx=self._scale, fy=self._scale)
        rgb_small = np.ascontiguousarray(small[:, :, ::-1])   # BGR → RGB

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        detections: list[FaceDetection] = []
        scale_inv = int(1.0 / self._scale)   # typically 4

        now = time.time()

        # Track which known records had a hit this frame (for consec counter)
        hit_names: set[str] = set()

        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            name     = "Unknown"
            title    = ""
            distance = 1.0
            is_known = False
            session  = ""

            if self._known:
                known_encodings = [r.encoding for r in self._known]
                matches   = face_recognition.compare_faces(
                    known_encodings, enc, tolerance=self._tolerance
                )
                distances = face_recognition.face_distance(known_encodings, enc)

                best_idx  = int(np.argmin(distances))
                if matches[best_idx]:
                    rec      = self._known[best_idx]
                    name     = rec.name
                    title    = rec.title
                    distance = float(distances[best_idx])
                    is_known = True
                    session  = f"face_{name.lower()}"
                    hit_names.add(name)

            # Scale bbox back to full resolution
            top    *= scale_inv
            right  *= scale_inv
            bottom *= scale_inv
            left   *= scale_inv

            if not is_known:
                session = self._unknown.session_id

            det = FaceDetection(
                name=name, title=title, distance=distance,
                bbox=(top, right, bottom, left),
                is_known=is_known, session_id=session,
            )
            detections.append(det)

            # ── Annotate frame ──────────────────────────────────
            color  = (0, 255, 0) if is_known else (0, 0, 255)
            label  = f"{title} {name}".strip() if title else name
            if is_known:
                label += f" ({distance:.2f})"

            cv2.rectangle(bgr_frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(bgr_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                bgr_frame, label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.75,
                (255, 255, 255), 1,
            )

        # ── Update consecutive counters + fire triggers ─────────
        self._update_and_trigger(detections, hit_names, bgr_frame, now)

        # Reset unknown session_id if no unknowns in frame
        # (so next unknown gets fresh session)
        any_unknown = any(not d.is_known for d in detections)
        if not any_unknown:
            if self._unknown.consec_frames > 0:
                self._unknown = _UnknownRecord()   # reset session

        return bgr_frame, detections

    # ── Consecutive-frame gating + cooldown ─────────────────────
    def _update_and_trigger(
        self,
        detections: list[FaceDetection],
        hit_names: set[str],
        frame: np.ndarray,
        now: float,
    ):
        # ── Known faces ─────────────────────────────────────────
        for rec in self._known:
            if rec.name in hit_names:
                rec.consec_frames += 1
            else:
                rec.consec_frames = 0
                continue

            if rec.consec_frames < self._min_frames:
                continue  # not yet stable

            if now - rec.last_trigger < self._cd_known:
                continue  # cooldown active

            rec.last_trigger  = now
            rec.consec_frames = 0   # reset after trigger

            # Find matching detection to pass to callback
            det = next((d for d in detections if d.name == rec.name), None)
            if det and self._trigger_cb:
                log.info(f"Trigger: KNOWN face '{rec.name}' (dist={det.distance:.3f})")
                self._trigger_cb(det, frame.copy())

        # ── Unknown persons ──────────────────────────────────────
        unknown_dets = [d for d in detections if not d.is_known]
        if unknown_dets:
            self._unknown.consec_frames += 1

            if (
                self._unknown.consec_frames >= self._min_frames
                and now - self._unknown.last_trigger >= self._cd_unknown
            ):
                self._unknown.last_trigger  = now
                self._unknown.consec_frames = 0

                log.info("Trigger: UNKNOWN person detected")
                if self._trigger_cb:
                    # Use the first unknown detection representative
                    self._trigger_cb(unknown_dets[0], frame.copy())
        else:
            self._unknown.consec_frames = 0
