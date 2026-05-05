"""
Video processing pipeline — ByteTrack + OSNet ReID.

Tracking  : YOLO model.track() with ByteTrack (single GPU pass, Kalman filter,
            Hungarian algorithm, lost-tracklet buffer).
ReID      : OSNet-x0_25 appearance embeddings extracted per person crop and
            stored per TrackedIndividual for image-to-person search.
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple
import os
import uuid
import logging
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from app.ai.aggregator import ViolationAggregator, ViolationRecord
from app.ai.reid import get_reid_service
from app.config import settings

logger = logging.getLogger(__name__)

# Absolute path to our bytetrack config — works regardless of CWD
_BACKEND_DIR = Path(__file__).parent.parent.parent  # backend/
BYTETRACK_YAML = str(_BACKEND_DIR / "bytetrack.yaml")

VIOLATION_CAPTURE_COOLDOWN = 2.0

BODY_PART_TO_VIOLATION = {
    'face': ('No Face Mask', ['face-mask', 'facemask', 'mask', 'face mask']),
    'foot': ('No Safety Boots', ['boots', 'shoes', 'safety-boots', 'safety boots']),
    'feet': ('No Safety Boots', ['boots', 'shoes', 'safety-boots', 'safety boots']),
    'hand': ('No Gloves', ['gloves', 'glove']),
    'hands': ('No Gloves', ['gloves', 'glove']),
}

GOGGLES_DETECTION_BUFFER = 2.0   # seconds before flagging No Goggles
HELMET_DETECTION_BUFFER = 2.0   # seconds before flagging No Helmet
VEST_DETECTION_BUFFER = 2.0     # seconds before flagging No Safety Vest
MASK_DETECTION_BUFFER = 2.0     # seconds before flagging No Face Mask
GLOVES_DETECTION_BUFFER = 2.0   # seconds before flagging No Gloves
BOOTS_DETECTION_BUFFER = 2.0    # seconds before flagging No Safety Boots

# Max time used to saturate the exposure confidence factor (5s → conf=1.0)
_CONF_SATURATION_S = 5.0

PPE_EQUIPMENT = ['helmet', 'face-mask', 'facemask', 'mask', 'glasses', 'goggles', 
                 'shoes', 'boots', 'safety-glasses', 'safety-vest', 'gloves', 'glove']


# PersonTracker removed — ByteTrack (via model.track) handles person tracking.


@dataclass
class ProcessingResultSimple:
    """Processing result."""
    success: bool
    total_frames: int
    processed_frames: int
    fps: float
    duration: float
    width: int
    height: int
    individual_profiles: dict
    violations: list
    person_worn_ppe: dict = None
    person_embeddings: dict = None   # {track_id: np.ndarray(512)} — OSNet ReID
    annotated_video_path: str = None
    error_message: Optional[str] = None


class VideoPipeline:
    """
    Pipeline using YOLO detection with configurable person tracking.
    
    Supports two tracking methods (configurable in config.py):
    - IOU: Position-based overlap matching (robust to appearance changes)
    - Cosine: Appearance-based matching (better re-identification)
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.model = YOLO(self.model_path)
        self.aggregator = ViolationAggregator()

        # ReID service (OSNet) — lazy-loaded, degrades gracefully if not installed
        self.reid = get_reid_service()
        # Best embedding per track_id: {track_id: np.ndarray(512)}
        self.person_embeddings: Dict[int, np.ndarray] = {}

        self.detection_interval_seconds = settings.DETECTION_INTERVAL_SECONDS
        self.frame_skip = settings.FRAME_SKIP

        self.is_processing = False
        self.progress = 0.0

        self.track_first_seen: Dict[int, int] = {}
        self.captured_violations: set = set()
        self.person_worn_ppe: Dict[int, set] = {}

        self.person_goggles_last_seen: Dict[int, float] = {}
        self.person_helmet_last_seen: Dict[int, float] = {}
        self.person_vest_last_seen: Dict[int, float] = {}
        self.person_mask_last_seen: Dict[int, float] = {}
        self.person_gloves_last_seen: Dict[int, float] = {}
        self.person_boots_last_seen: Dict[int, float] = {}

        self.detected_equipment: List[dict] = []
        self.violation_display_threshold = settings.VIOLATION_DISPLAY_THRESHOLD
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

        import torch
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        logger.info(
            f"Pipeline initialised on device={self.device} | "
            f"tracker=ByteTrack | ReID={'enabled' if self.reid.available else 'disabled'}"
        )
        logger.info(f"Model classes: {self.class_names}")
    
    def soft_reset(self):
        """Reset tracking state WITHOUT reloading the model.
        
        Use this between webcam sessions — model reload is too slow (~10s) and
        would drop the WebSocket. ByteTrack IDs simply restart from 1 each session.
        """
        self.aggregator.reset()
        self.is_processing = False
        self.progress = 0.0
        self.track_first_seen = {}
        self.captured_violations = set()
        self.person_worn_ppe = {}
        self.person_embeddings = {}
        self.person_goggles_last_seen = {}
        self.person_helmet_last_seen = {}
        self.person_vest_last_seen = {}
        self.person_mask_last_seen = {}
        self.person_gloves_last_seen = {}
        self.person_boots_last_seen = {}
        self.detected_equipment = []
        logger.info("Pipeline soft-reset (model kept in memory)")

    def reset(self):
        """Reset for new video. Re-creates YOLO model to flush ByteTrack internal state."""
        self.soft_reset()
        # Re-instantiate to flush ByteTrack's Kalman filter state
        self.model = YOLO(self.model_path)
        logger.info("Pipeline full-reset (model reloaded)")
    
    def _is_ppe_equipment(self, class_name: str) -> bool:
        """Check if class is PPE equipment (indicates compliance)."""
        class_lower = class_name.lower()
        return any(ppe in class_lower or class_lower in ppe for ppe in PPE_EQUIPMENT)
    
    def _can_capture(self, track_id: int, vtype: str, ts: float) -> bool:
        """Check if we should capture - only one snapshot per person per violation type."""
        key = (track_id, vtype)
        if key in self.captured_violations:
            return False  
        self.captured_violations.add(key)
        return True
    
    def _should_skip_violation(self, track_id: int, violation_type: str) -> bool:
        """
        Check if violation should be skipped because person has worn corresponding PPE.
        
        If a person is detected wearing PPE at any point, they don't get violations
        for that missing PPE type.
        
        NOTE: 'No Goggles', 'No Gloves', 'No Safety Boots', and 'No Face Mask' are handled 
        dynamically per-frame, not by cumulative tracking, so they're excluded from this skip logic.
        """
        DYNAMIC_VIOLATIONS = ['No Helmet', 'No Goggles', 'No Gloves', 'No Safety Boots', 'No Face Mask', 'No Safety Vest']
        if violation_type in DYNAMIC_VIOLATIONS:
            return False
        
        worn_ppe = self.person_worn_ppe.get(track_id, set())
        required_ppe = VIOLATION_TO_PPE.get(violation_type, [])
        
        for ppe in required_ppe:
            for worn_item in worn_ppe:
                if ppe in worn_item or worn_item in ppe:
                    logger.debug(f"Skipping {violation_type} for Person-{track_id}: detected wearing {worn_item}")
                    return True
        return False
    
    def _record_person_ppe(self, track_id: int, ppe_type: str):
        """Record that a person has been detected wearing a specific PPE item."""
        if track_id not in self.person_worn_ppe:
            self.person_worn_ppe[track_id] = set()
        
        ppe_lower = ppe_type.lower()
        if ppe_lower not in self.person_worn_ppe[track_id]:
            self.person_worn_ppe[track_id].add(ppe_lower)
            logger.info(f"PPE Detected: Person-{track_id} wearing {ppe_type}")
    
    def _convert_to_browser_compatible(self, input_path: str, output_path: str) -> bool:
        """Convert video to browser-compatible format using ffmpeg."""
        try:
            if not shutil.which('ffmpeg'):
                logger.warning("ffmpeg not found, video may not play in browser")
                return False
            
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except Exception as e:
            logger.error(f"ffmpeg conversion failed: {e}")
            return False
    
    def process_video_sync(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> ProcessingResultSimple:
        """Process video with YOLO native tracking."""
        self.reset()
        self.is_processing = True
        
        logger.info(f"Processing with YOLO native tracking: {video_path}")
        
        video_id = uuid.uuid4().hex[:8]
        temp_output = os.path.join(settings.UPLOAD_DIR, f"temp_{video_id}.mp4")
        final_output = os.path.join(settings.UPLOAD_DIR, f"annotated_{video_id}.mp4")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
            
            codecs_to_try = [
                ('avc1', '.mp4'),   
                ('H264', '.mp4'),   
                ('X264', '.mp4'),   
                ('mp4v', '.mp4'),   
            ]
            
            out = None
            for codec, ext in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_path = os.path.join(settings.UPLOAD_DIR, f"annotated_{video_id}{ext}")
                    out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        final_output = test_path
                        logger.info(f"Using codec: {codec}")
                        break
                    out.release()
                except:
                    continue
            
            if out is None or not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                final_output = os.path.join(settings.UPLOAD_DIR, f"annotated_{video_id}.mp4")
                out = cv2.VideoWriter(final_output, fourcc, fps, (width, height))
                logger.warning("Using mp4v codec - video may not play in browser")
            
            self.aggregator.fps = fps
            
            if self.detection_interval_seconds > 0:
                effective_frame_skip = max(1, int(fps * self.detection_interval_seconds))
                logger.info(f"Using time-based detection: every {self.detection_interval_seconds}s = every {effective_frame_skip} frames")
            else:
                effective_frame_skip = self.frame_skip
                logger.info(f"Using frame-based detection: every {effective_frame_skip} frames")
            
            frame_num = 0
            processed = 0
            
            last_annotations = {
                'persons': [],      
                'violations': [],   
                'ppe_items': [],    
                'timestamp': 0.0,
                'total_violations': 0
            }
            
            def draw_annotations_on_frame(frame, annotations, current_timestamp):
                """Re-draw stored annotations on a frame for smooth visualization."""
                annotated = frame.copy()
                
                for person_bbox, track_id in annotations['persons']:
                    px1, py1, px2, py2 = [int(c) for c in person_bbox]
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    label = f"Person-{track_id}"
                    cv2.putText(annotated, label, (px1, py1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                for ppe_bbox, cls_name in annotations['ppe_items']:
                    ex1, ey1, ex2, ey2 = [int(c) for c in ppe_bbox]
                    cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
                    cv2.putText(annotated, cls_name, (ex1, ey1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                for vbox, vtype, track_id in annotations['violations']:
                    vx1, vy1, vx2, vy2 = [int(c) for c in vbox]
                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                    label = f"Person-{track_id}: {vtype}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated, (vx1, vy1 - label_size[1] - 10),
                                 (vx1 + label_size[0] + 10, vy1), (0, 0, 255), -1)
                    cv2.putText(annotated, label, (vx1 + 5, vy1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                num_persons = len(set(p[1] for p in annotations['persons']))
                info = f"Time: {current_timestamp:.1f}s | Persons: {num_persons} | Violations: {annotations['total_violations']}"
                cv2.rectangle(annotated, (5, 5), (500, 40), (0, 0, 0), -1)
                cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                return annotated
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_timestamp = frame_num / fps
                
                if frame_num % effective_frame_skip == 0:
                    annotated = self._process_frame_with_tracking(
                        frame, frame_num, video_id, fps, last_annotations
                    )
                    processed += 1
                    self.progress = (frame_num / total_frames) * 100
                    if progress_callback:
                        try:
                            progress_callback(self.progress)
                        except:
                            pass
                else:
                    annotated = draw_annotations_on_frame(frame, last_annotations, current_timestamp)
                
                out.write(annotated)
                frame_num += 1
            
            cap.release()
            out.release()
            
            annotated_path = final_output
            
            profiles = self.aggregator.get_all_profiles()
            violations = []
            for p in profiles.values():
                for v in p.violations:
                    violations.append({
                        "track_id": p.track_id,
                        "person_name": f"Person-{p.track_id}",
                        "type": v.violation_type,
                        "confidence": v.confidence,
                        "frame": v.frame_number,
                        "timestamp": v.timestamp,
                        "bbox": v.bbox,
                        "image_path": v.image_path
                    })
            
            logger.info(f"Complete: {len(violations)} violations, {len(profiles)} persons")
            
            for tid, profile in profiles.items():
                if profile.violation_count > 0:
                    types_str = ', '.join(f'{t}:{c}' for t, c in profile.violation_types.items())
                    logger.info(f"Person-{tid}: {profile.violation_count} violations ({types_str})")
            
            return ProcessingResultSimple(
                success=True, total_frames=total_frames, processed_frames=processed,
                fps=fps, duration=duration, width=width, height=height,
                individual_profiles={tid: p for tid, p in profiles.items()},
                violations=violations,
                person_worn_ppe=self.person_worn_ppe.copy(),
                person_embeddings=self.person_embeddings.copy(),
                annotated_video_path=annotated_path
            )
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return ProcessingResultSimple(
                success=False, total_frames=0, processed_frames=0,
                fps=0, duration=0, width=0, height=0,
                individual_profiles={}, violations=[], error_message=str(e)
            )
        finally:
            self.is_processing = False
    
    def _process_frame_with_tracking(
        self, frame: np.ndarray, frame_num: int, video_id: str, fps: float,
        last_annotations: dict = None
    ) -> np.ndarray:
        """
        Process frame with custom IOU-based person tracking.
        
        1. Detect all objects using YOLO
        2. Use custom PersonTracker for consistent person IDs
        3. Associate violations with nearest tracked person
        4. Record detected PPE equipment
        5. Store annotation data in last_annotations for smooth skipped-frame rendering
        """
        timestamp = frame_num / fps

        # ── ByteTrack: single-pass detection + tracking ──────────────────────
        # persist=True keeps Kalman filter state across calls (crucial for consistent IDs).
        # We use an absolute path to bytetrack.yaml so it resolves regardless of CWD.
        results = self.model.track(
            frame,
            persist=True,
            tracker=BYTETRACK_YAML,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            device=self.device,
            verbose=False,
            classes=None,
        )
        
        annotated = frame.copy()
        
        if not results or len(results) == 0:
            return annotated
        
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return annotated
        
        person_detections = []  
        violations = [] 
        ppe_items = [] 
        body_parts = [] 
        
        NO_PPE_KEYWORDS = {
            'no-mask': 'No Face Mask', 'no_mask': 'No Face Mask', 'no mask': 'No Face Mask', 'nomask': 'No Face Mask',
            'no-goggles': 'No Goggles', 'no_goggles': 'No Goggles', 'no goggles': 'No Goggles', 'nogoggles': 'No Goggles',
            'no-glasses': 'No Goggles', 'no_glasses': 'No Goggles', 'no glasses': 'No Goggles',
            'no-helmet': 'No Helmet', 'no_helmet': 'No Helmet', 'no helmet': 'No Helmet', 'nohelmet': 'No Helmet',
            'no-boots': 'No Safety Boots', 'no_boots': 'No Safety Boots', 'no boots': 'No Safety Boots',
            'no-gloves': 'No Gloves', 'no_gloves': 'No Gloves', 'no gloves': 'No Gloves',
            'no-vest': 'No Safety Vest', 'no_vest': 'No Safety Vest', 'no vest': 'No Safety Vest',
        }
        
        BODY_PART_TO_VIOLATION = {
            'face': ('No Face Mask', ['face-mask', 'facemask', 'mask', 'face mask']),
            'eyes': ('No Goggles', ['glasses', 'goggles', 'safety-glasses', 'eye protection']),
            'eye': ('No Goggles', ['glasses', 'goggles', 'safety-glasses', 'eye protection']),
            'head': ('No Helmet', ['helmet', 'hard hat', 'hardhat']),
            'hand': ('No Gloves', ['gloves', 'glove']),
            'hands': ('No Gloves', ['gloves', 'glove']),
            'foot': ('No Safety Boots', ['boots', 'shoes', 'safety-boots']),
            'feet': ('No Safety Boots', ['boots', 'shoes', 'safety-boots']),
        }
        
        # ByteTrack assigns IDs only to the tracked class — we track 'person'.
        # Non-person classes (PPE, violations) come from boxes without .id.
        track_ids_tensor = boxes.id  # may be None if no tracks yet

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = [int(c) for c in xyxy]
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            bbox = (x1, y1, x2, y2)
            cls_lower = cls_name.lower()

            if cls_lower == 'person':
                # ByteTrack track ID (int) or fall back to sequential counter
                if track_ids_tensor is not None:
                    bt_id = int(track_ids_tensor[i].cpu().numpy())
                else:
                    bt_id = i + 1
                person_detections.append((bbox, conf, bt_id))
            else:
                violation_type = None
                for keyword, vtype in NO_PPE_KEYWORDS.items():
                    if keyword in cls_lower:
                        violation_type = vtype
                        break

                if violation_type:
                    violations.append((bbox, violation_type, conf))
                elif cls_lower in BODY_PART_TO_VIOLATION:
                    body_parts.append((bbox, cls_lower, conf))
                elif self._is_ppe_equipment(cls_name):
                    ppe_items.append((bbox, cls_name, conf))
                    self.detected_equipment.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'type': cls_name,
                        'confidence': conf,
                        'bbox': bbox
                    })

        # persons is now List[(bbox, track_id, conf)] — same shape as before
        persons = [(bbox, tid, conf) for bbox, conf, tid in person_detections]
        
        if frame_num % 30 == 0:
            logger.info(f"Frame {frame_num}: {len(persons)} persons (tracked), {len(violations)} violations, {len(ppe_items)} PPE")
        
        def find_closest_person(vbox):
            """Find person whose bbox is closest/overlapping with violation bbox."""
            vx1, vy1, vx2, vy2 = vbox
            vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2  
            
            best_person = None
            best_dist = float('inf')
            
            for person_bbox, person_tid, person_conf in persons:
                px1, py1, px2, py2 = person_bbox
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                dist = ((vcx - pcx) ** 2 + (vcy - pcy) ** 2) ** 0.5
                
                if vx1 >= px1 - 50 and vx2 <= px2 + 50 and vy1 >= py1 - 50 and vy2 <= py2 + 50:
                    dist = 0  
                
                if dist < best_dist:
                    best_dist = dist
                    best_person = (person_bbox, person_tid)
            
            return best_person
        
        for person_bbox, track_id, person_conf in persons:
            px1, py1, px2, py2 = person_bbox

            if track_id not in self.track_first_seen:
                self.track_first_seen[track_id] = frame_num
                logger.info(f"ByteTrack: New Person-{track_id}")

            # ── OSNet ReID: extract embedding every ~30 frames to save CPU ──
            if self.reid.available and frame_num % 30 == 0:
                emb = self.reid.extract(frame, person_bbox)
                if emb is not None:
                    if track_id not in self.person_embeddings:
                        self.person_embeddings[track_id] = emb
                    else:
                        # Running average for more stable representation
                        self.person_embeddings[track_id] = (
                            0.7 * self.person_embeddings[track_id] + 0.3 * emb
                        )

            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            label = f"Person-{track_id}"
            cv2.putText(annotated, label, (px1, py1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        current_frame_ppe = {}  
        
        for ppe_bbox, cls_name, conf in ppe_items:
            ex1, ey1, ex2, ey2 = ppe_bbox
            cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
            cv2.putText(annotated, cls_name, (ex1, ey1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            ppe_cx, ppe_cy = (ex1 + ex2) / 2, (ey1 + ey2) / 2  
            
            for person_bbox, person_tid, person_conf in persons:
                px1, py1, px2, py2 = person_bbox
                tolerance = 30  
                if (px1 - tolerance <= ppe_cx <= px2 + tolerance and 
                    py1 - tolerance <= ppe_cy <= py2 + tolerance):
                    self._record_person_ppe(person_tid, cls_name)
                    if person_tid not in current_frame_ppe:
                        current_frame_ppe[person_tid] = set()
                    current_frame_ppe[person_tid].add(cls_name.lower())
                    break  
        

        GOGGLES_CLASSES = ['glasses', 'goggles', 'safety-glasses', 'safety glasses', 'eye protection', 'eyeglasses']
        
        
        def is_face_detected_for_person(person_bbox):
            """Return face detection confidence if 'face' body part is detected near this person, else 0.0."""
            px1, py1, px2, py2 = person_bbox
            for body_bbox, body_part, bp_conf in body_parts:
                if body_part == 'face':
                    bx1, by1, bx2, by2 = body_bbox
                    face_cx, face_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    tolerance = 50
                    if (px1 - tolerance <= face_cx <= px2 + tolerance and
                        py1 - tolerance <= face_cy <= py2 + tolerance):
                        return bp_conf
            return 0.0
        
        
        def is_hand_detected_for_person(person_bbox):
            """Return hand detection confidence if 'hand'/'hands' detected near this person, else 0.0."""
            px1, py1, px2, py2 = person_bbox
            for body_bbox, body_part, bp_conf in body_parts:
                if body_part in ['hand', 'hands']:
                    bx1, by1, bx2, by2 = body_bbox
                    hand_cx, hand_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    tolerance = 50
                    if (px1 - tolerance <= hand_cx <= px2 + tolerance and
                        py1 - tolerance <= hand_cy <= py2 + tolerance):
                        return bp_conf
            return 0.0

        
        def is_foot_detected_for_person(person_bbox):
            """Return foot detection confidence if 'foot'/'feet' detected near this person, else 0.0."""
            px1, py1, px2, py2 = person_bbox
            for body_bbox, body_part, bp_conf in body_parts:
                if body_part in ['foot', 'feet', 'shoe', 'shoes']:
                    bx1, by1, bx2, by2 = body_bbox
                    foot_cx, foot_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    tolerance = 50
                    if (px1 - tolerance <= foot_cx <= px2 + tolerance and
                        py1 - tolerance <= foot_cy <= py2 + tolerance):
                        return bp_conf
            return 0.0
        
        
        def is_head_detected_for_person(person_bbox):
            """Return head detection confidence if 'head'/'face' detected near this person, else 0.0."""
            px1, py1, px2, py2 = person_bbox
            for body_bbox, body_part, bp_conf in body_parts:
                if body_part in ['head', 'face']:
                    bx1, by1, bx2, by2 = body_bbox
                    head_cx, head_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    tolerance = 50
                    if (px1 - tolerance <= head_cx <= px2 + tolerance and
                        py1 - tolerance <= head_cy <= py2 + tolerance):
                        return bp_conf
            return 0.0
        
        for person_bbox, track_id, person_conf in persons:
            
            face_conf = is_face_detected_for_person(person_bbox)
            if face_conf == 0.0:
                
                if track_id in self.person_goggles_last_seen:
                    del self.person_goggles_last_seen[track_id]
                continue
            
            
            person_ppe = current_frame_ppe.get(track_id, set())
            
            
            goggles_detected = False
            for ppe_item in person_ppe:
                for goggles_type in GOGGLES_CLASSES:
                    if goggles_type in ppe_item or ppe_item in goggles_type:
                        goggles_detected = True
                        break
                if goggles_detected:
                    break
            
            if goggles_detected:
                
                self.person_goggles_last_seen[track_id] = timestamp
            else:
                
                last_seen = self.person_goggles_last_seen.get(track_id, -999)  
                
                if last_seen == -999:
                    
                    self.person_goggles_last_seen[track_id] = timestamp
                else:
                    time_without_goggles = timestamp - last_seen
                    if time_without_goggles >= GOGGLES_DETECTION_BUFFER:
                        px1, py1, px2, py2 = person_bbox
                        head_height = int((py2 - py1) * 0.3)
                        goggles_bbox = (px1, py1, px2, py1 + head_height)
                        # Confidence = blend of: face detection conf (40%)
                        #              + time-exposure factor (60%, saturates at _CONF_SATURATION_S)
                        exposure_factor = min(time_without_goggles / _CONF_SATURATION_S, 1.0)
                        viol_conf = round(0.4 * face_conf + 0.6 * exposure_factor, 3)
                        violations.append((goggles_bbox, 'No Goggles', viol_conf))
                        if frame_num % 30 == 0:
                            logger.info(f"No Goggles violation for Person-{track_id} (face_conf={face_conf:.2f}, absent {time_without_goggles:.1f}s, conf={viol_conf:.2f})")
        
        
        HELMET_CLASSES = ['helmet', 'hard hat', 'hardhat', 'safety helmet']
        
        for person_bbox, track_id, person_conf in persons:
            
            head_conf = is_head_detected_for_person(person_bbox)
            if head_conf == 0.0:
                
                if track_id in self.person_helmet_last_seen:
                    del self.person_helmet_last_seen[track_id]
                continue
            person_ppe = current_frame_ppe.get(track_id, set())
            
            helmet_detected = False
            for ppe_item in person_ppe:
                for helmet_type in HELMET_CLASSES:
                    if helmet_type in ppe_item or ppe_item in helmet_type:
                        helmet_detected = True
                        break
                if helmet_detected:
                    break
            
            if helmet_detected:
                
                self.person_helmet_last_seen[track_id] = timestamp
            else:
                
                last_seen = self.person_helmet_last_seen.get(track_id, -999)  
                
                if last_seen == -999:
                    
                    self.person_helmet_last_seen[track_id] = timestamp
                else:
                    time_without_helmet = timestamp - last_seen
                    if time_without_helmet >= HELMET_DETECTION_BUFFER:
                        px1, py1, px2, py2 = person_bbox
                        head_height = int((py2 - py1) * 0.4)
                        helmet_bbox = (px1, py1, px2, py1 + head_height)
                        exposure_factor = min(time_without_helmet / _CONF_SATURATION_S, 1.0)
                        viol_conf = round(0.4 * head_conf + 0.6 * exposure_factor, 3)
                        violations.append((helmet_bbox, 'No Helmet', viol_conf))
                        if frame_num % 30 == 0:
                            logger.info(f"No Helmet violation for Person-{track_id} (head_conf={head_conf:.2f}, absent {time_without_helmet:.1f}s, conf={viol_conf:.2f})")
        
        
        VEST_CLASSES = ['vest', 'safety-vest', 'safety vest', 'hi-vis', 'high-visibility', 'reflective vest']
        
        for person_bbox, track_id, person_conf in persons:
            person_ppe = current_frame_ppe.get(track_id, set())
            
            vest_detected = False
            for ppe_item in person_ppe:
                for vest_type in VEST_CLASSES:
                    if vest_type in ppe_item or ppe_item in vest_type:
                        vest_detected = True
                        break
                if vest_detected:
                    break
            
            if vest_detected:
                
                self.person_vest_last_seen[track_id] = timestamp
            else:
                
                last_seen = self.person_vest_last_seen.get(track_id, -999)
                
                if last_seen == -999:
                    
                    self.person_vest_last_seen[track_id] = timestamp
                else:
                    time_without_vest = timestamp - last_seen
                    if time_without_vest >= VEST_DETECTION_BUFFER:
                        px1, py1, px2, py2 = person_bbox
                        person_height = py2 - py1
                        torso_top = py1 + int(person_height * 0.2)
                        torso_bottom = py1 + int(person_height * 0.8)
                        vest_bbox = (px1, torso_top, px2, torso_bottom)
                        exposure_factor = min(time_without_vest / _CONF_SATURATION_S, 1.0)
                        viol_conf = round(0.4 * person_conf + 0.6 * exposure_factor, 3)
                        violations.append((vest_bbox, 'No Safety Vest', viol_conf))
                        if frame_num % 30 == 0:
                            logger.info(f"No Safety Vest violation for Person-{track_id} (absent {time_without_vest:.1f}s, conf={viol_conf:.2f})")
        
        
        MASK_CLASSES = ['mask', 'face-mask', 'facemask', 'face mask', 'surgical mask', 'n95']
        
        for person_bbox, track_id, person_conf in persons:
            face_conf_mask = is_face_detected_for_person(person_bbox)
            if face_conf_mask == 0.0:
                
                if track_id in self.person_mask_last_seen:
                    del self.person_mask_last_seen[track_id]
                continue
            
            person_ppe = current_frame_ppe.get(track_id, set())
            
            mask_detected = False
            for ppe_item in person_ppe:
                for mask_type in MASK_CLASSES:
                    if mask_type in ppe_item or ppe_item in mask_type:
                        mask_detected = True
                        break
                if mask_detected:
                    break
            
            if mask_detected:
                
                self.person_mask_last_seen[track_id] = timestamp
            else:
                
                last_seen = self.person_mask_last_seen.get(track_id, -999)
                
                if last_seen == -999:
                    
                    self.person_mask_last_seen[track_id] = timestamp
                else:
                    time_without_mask = timestamp - last_seen
                    if time_without_mask >= MASK_DETECTION_BUFFER:
                        px1, py1, px2, py2 = person_bbox
                        face_height = int((py2 - py1) * 0.35)
                        mask_bbox = (px1, py1, px2, py1 + face_height)
                        exposure_factor = min(time_without_mask / _CONF_SATURATION_S, 1.0)
                        viol_conf = round(0.4 * face_conf_mask + 0.6 * exposure_factor, 3)
                        violations.append((mask_bbox, 'No Face Mask', viol_conf))
                        if frame_num % 30 == 0:
                            logger.info(f"No Face Mask violation for Person-{track_id} (face_conf={face_conf_mask:.2f}, absent {time_without_mask:.1f}s, conf={viol_conf:.2f})")
        
        
        GLOVES_CLASSES = ['gloves', 'glove', 'hand protection', 'work gloves', 'safety gloves']
        
        for person_bbox, track_id, person_conf in persons:
            hand_conf = is_hand_detected_for_person(person_bbox)
            if hand_conf == 0.0:
                
                if track_id in self.person_gloves_last_seen:
                    del self.person_gloves_last_seen[track_id]
                continue
            person_ppe = current_frame_ppe.get(track_id, set())
            
            gloves_detected = False
            for ppe_item in person_ppe:
                for gloves_type in GLOVES_CLASSES:
                    if gloves_type in ppe_item or ppe_item in gloves_type:
                        gloves_detected = True
                        break
                if gloves_detected:
                    break
            
            if gloves_detected:
                
                self.person_gloves_last_seen[track_id] = timestamp
            else:
                
                last_seen = self.person_gloves_last_seen.get(track_id, -999)
                
                if last_seen == -999:
                    
                    self.person_gloves_last_seen[track_id] = timestamp
                else:
                    time_without_gloves = timestamp - last_seen
                    if time_without_gloves >= GLOVES_DETECTION_BUFFER:
                        px1, py1, px2, py2 = person_bbox
                        person_height = py2 - py1
                        hands_top = py1 + int(person_height * 0.4)
                        hands_bottom = py1 + int(person_height * 0.7)
                        gloves_bbox = (px1, hands_top, px2, hands_bottom)
                        exposure_factor = min(time_without_gloves / _CONF_SATURATION_S, 1.0)
                        viol_conf = round(0.4 * hand_conf + 0.6 * exposure_factor, 3)
                        violations.append((gloves_bbox, 'No Gloves', viol_conf))
                        if frame_num % 30 == 0:
                            logger.info(f"No Gloves violation for Person-{track_id} (hand_conf={hand_conf:.2f}, absent {time_without_gloves:.1f}s, conf={viol_conf:.2f})")
        
        
        BOOTS_CLASSES = ['boots', 'shoes', 'safety-boots', 'safety boots', 'safety shoes', 'work boots', 'footwear']
        
        for person_bbox, track_id, person_conf in persons:
            foot_conf = is_foot_detected_for_person(person_bbox)
            if foot_conf == 0.0:
                
                if track_id in self.person_boots_last_seen:
                    del self.person_boots_last_seen[track_id]
                continue
            person_ppe = current_frame_ppe.get(track_id, set())
            
            
            boots_detected = False
            for ppe_item in person_ppe:
                for boots_type in BOOTS_CLASSES:
                    if boots_type in ppe_item or ppe_item in boots_type:
                        boots_detected = True
                        break
                if boots_detected:
                    break
            
            if boots_detected:
                # Boots detected - update last seen time, no violation
                self.person_boots_last_seen[track_id] = timestamp
            else:
                # Boots NOT detected - check if buffer has passed
                last_seen = self.person_boots_last_seen.get(track_id, -999)
                
                if last_seen == -999:
                    # First time seeing this person without boots - start the timer
                    self.person_boots_last_seen[track_id] = timestamp
                else:
                    time_without_boots = timestamp - last_seen
                    if time_without_boots >= BOOTS_DETECTION_BUFFER:
                        # Use lower 30% of person bbox for boots (feet area)
                        px1, py1, px2, py2 = person_bbox
                        person_height = py2 - py1
                        feet_top = py1 + int(person_height * 0.7)
                        boots_bbox = (px1, feet_top, px2, py2)
                        exposure_factor = min(time_without_boots / _CONF_SATURATION_S, 1.0)
                        viol_conf = round(0.4 * foot_conf + 0.6 * exposure_factor, 3)
                        violations.append((boots_bbox, 'No Safety Boots', viol_conf))
                        if frame_num % 30 == 0:
                            logger.info(f"No Safety Boots violation for Person-{track_id} (foot_conf={foot_conf:.2f}, absent {time_without_boots:.1f}s, conf={viol_conf:.2f})")
        
        # NOTE: Legacy body-part based violation loop removed.
        # We now use the robust, buffered person-based checks above (Helmet, Vest, Mask, Gloves, Boots).
        # Body parts are only used as prerequisites for those checks.
        
        # THEN Process violations - skip if person has worn corresponding PPE
        frame_drawn_violations = []  # Track violations drawn in this frame for smooth skipped-frame rendering
        
        for vbox, vtype, conf in violations:
            vx1, vy1, vx2, vy2 = vbox
            
            # Find which person this violation belongs to
            person_bbox = None
            closest = find_closest_person(vbox)
            if closest:
                person_bbox, track_id = closest
            else:
                # No person found, use fallback ID
                track_id = 1
                if track_id not in self.track_first_seen:
                    self.track_first_seen[track_id] = frame_num
            
            # Check if this violation should be skipped because person has worn corresponding PPE
            if self._should_skip_violation(track_id, vtype):
                continue  # Skip this violation - person has worn the required PPE
            
            # Draw violation box (RED)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
            if vtype == 'No Goggles':
                logger.info(f"Drawing No Goggles box at ({vx1},{vy1})-({vx2},{vy2}) for Person-{track_id}")
            
            # Track this drawn violation for skipped-frame rendering
            frame_drawn_violations.append((vbox, vtype, track_id))
            
            # Violation label
            label = f"Person-{track_id}: {vtype}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (vx1, vy1 - label_size[1] - 10),
                         (vx1 + label_size[0] + 10, vy1), (0, 0, 255), -1)
            cv2.putText(annotated, label, (vx1 + 5, vy1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update aggregator with PERSON's track ID
            self.aggregator.update_individual(
                track_id=track_id,
                frame_number=frame_num,
                first_seen_frame=self.track_first_seen.get(track_id, frame_num)
            )
            
            # Capture violation image with cooldown
            # ONLY capture if confidence meets the display threshold
            if conf >= self.violation_display_threshold and self._can_capture(track_id, vtype, timestamp):
                bbox = (float(vx1), float(vy1), float(vx2), float(vy2))
                img_path = self._save_image(frame, bbox, video_id, frame_num, vtype, track_id, person_bbox)
                
                record = ViolationRecord(
                    violation_type=vtype,
                    confidence=conf,
                    frame_number=frame_num,
                    timestamp=timestamp,
                    bbox=bbox,
                    image_path=img_path
                )
                self.aggregator.profiles[track_id].add_violation(record)
                logger.info(f"VIOLATION: Person-{track_id} - {vtype}")
        
        # Frame info overlay
        active_persons = len([p for p in persons])
        total_violations = sum(p.violation_count for p in self.aggregator.profiles.values())
        info = f"Time: {timestamp:.1f}s | Persons: {len(set(p[1] for p in persons))} | Violations: {total_violations}"
        cv2.rectangle(annotated, (5, 5), (500, 40), (0, 0, 0), -1)
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store annotation data for smooth skipped-frame rendering
        if last_annotations is not None:
            last_annotations['persons'] = [(p_bbox, p_tid) for p_bbox, p_tid, _ in persons]
            last_annotations['ppe_items'] = [(p_bbox, cls) for p_bbox, cls, _ in ppe_items]
            # Store violations that were drawn (not skipped)
            last_annotations['violations'] = frame_drawn_violations
            last_annotations['timestamp'] = timestamp
            last_annotations['total_violations'] = total_violations
        
        return annotated
    
    def _save_image(self, frame, bbox, video_id, frame_num, vtype, track_id, person_bbox=None, target_dir=None) -> str:
        """Save violation snapshot with highlighting and min size."""
        try:
            h, w = frame.shape[:2]
            img = frame.copy()
            
            # Use default violation dir if none provided
            if target_dir is None:
                target_dir = settings.VIOLATIONS_IMG_DIR
            
            # Ensure directory exists
            os.makedirs(target_dir, exist_ok=True)
            
            # Draw violation box (RED) on the image
            vx1, vy1, vx2, vy2 = [int(c) for c in bbox]
            cv2.rectangle(img, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
            
            # Determine initial crop area
            if person_bbox:
                # Crop the person
                cx1, cy1, cx2, cy2 = [int(c) for c in person_bbox]
                # Add 10% padding around person
                pw, ph = cx2 - cx1, cy2 - cy1
                pad_x, pad_y = int(pw * 0.1), int(ph * 0.1)
                cx1, cy1 = max(0, cx1 - pad_x), max(0, cy1 - pad_y)
                cx2, cy2 = min(w, cx2 + pad_x), min(h, cy2 + pad_y)
            else:
                # Fallback: Crop the violation box with 50% padding
                cx1, cy1, cx2, cy2 = vx1, vy1, vx2, vy2
                pw, ph = cx2 - cx1, cy2 - cy1
                pad_x, pad_y = int(pw * 0.5), int(ph * 0.5)
                cx1, cy1 = max(0, cx1 - pad_x), max(0, cy1 - pad_y)
                cx2, cy2 = min(w, cx2 + pad_x), min(h, cy2 + pad_y)
            
            # Enforce Minimum Size (200x200)
            target_w, target_h = 200, 200
            crop_w, crop_h = cx2 - cx1, cy2 - cy1
            
            if crop_w < target_w:
                diff = target_w - crop_w
                cx1 = max(0, cx1 - diff // 2)
                cx2 = min(w, cx2 + (diff - diff // 2))
                
            if crop_h < target_h:
                diff = target_h - crop_h
                cy1 = max(0, cy1 - diff // 2)
                cy2 = min(h, cy2 + (diff - diff // 2))
                
            # Perform Crop
            crop_img = img[cy1:cy2, cx1:cx2].copy()
            
            safe = vtype.replace(" ", "_").lower()
            fname = f"p{track_id}_{safe}_{video_id}_{frame_num}.jpg"
            path = os.path.join(target_dir, fname)
            cv2.imwrite(path, crop_img)
            
            # Return path relative to project root or accessible URL
            if target_dir == settings.VIOLATIONS_IMG_DIR:
                return f"/violation_images/{fname}"
            else:
                return f"/{target_dir}/{fname}"
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return None
    
    def get_progress(self) -> float:
        return self.progress
    
    def is_active(self) -> bool:
        return self.is_processing
