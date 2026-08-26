import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

def print_log(msg):
    print(msg, flush=True)

t0 = time.time()
print_log("[1/3] Initializing PyTorch, OpenCV, and RF-DETR engine...")
print_log("      (Please wait while PyTorch initializes CUDA/CPU weights...)")

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from deep_sort_realtime.deepsort_tracker import DeepSort
print_log(f"      [OK] Core libraries loaded in {time.time() - t0:.1f} seconds!")



# ---------------------------------------------------------------------------
# Configuration — tune these for your drone / camera setup
# ---------------------------------------------------------------------------
# SET THIS TO TRUE to test on yourself via webcam!
# If True, loads the standard COCO pre-trained model which recognizes sitting people.
# If False, loads your custom drone checkpoint (which only recognizes tiny standing people from above).
WEBCAM_TEST_MODE = True

# Capture resolution: 480p for drone video link bandwidth
CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480

# Detection threshold: lowered for small/distant humans at altitude.
# At 50 m with a zoomed 480p feed, humans may only be 20–40 px tall and
# produce lower-confidence detections than a close-up indoor webcam.
# We use a higher threshold for webcam testing to avoid false positives.
DETECTION_THRESHOLD = 0.5 if WEBCAM_TEST_MODE else 0.15

# Display: upscale the 480p feed for the operator's monitor so bounding
# boxes and labels are easy to read and click.  Set to 1.0 to disable.
DISPLAY_SCALE = 2.0  # 640×480 → 1280×960 on screen

# ByteTrack tuning for drone motion at ~30 fps:
#   track_activation_threshold : 0.10  — well below DETECTION_THRESHOLD so
#       ByteTrack's second-stage matching never internally discards a
#       detection we already passed it.  Critical for low-confidence
#       detections from altitude.
#   lost_track_buffer : 60 frames — ~2 s at 30 fps.  Drone vibration,
#       gimbal jitter, and momentary occlusions cause more frequent
#       detection dropouts than a static indoor camera; a longer buffer
#       keeps IDs alive through brief gaps.
#   minimum_matching_threshold : 0.4 — IoU gate.  Drone ego-motion causes
#       larger frame-to-frame displacement of bounding boxes than a
#       stationary webcam; lower IoU gate allows the Kalman-predicted
#       position to match detections even when overlap is only ~40 %.
#   minimum_consecutive_frames : 2 — require 2 consecutive detections
#       before promoting a tentative track to active.  At the lower
#       detection threshold (0.15) this prevents transient false positives
#       from becoming visible tracks without adding noticeable latency
#       (just 1 frame delay at 30 fps).
TRACKER_ACTIVATION_THRESHOLD  = 0.10
TRACKER_LOST_BUFFER           = 60
TRACKER_MATCHING_THRESHOLD    = 0.4
TRACKER_MIN_CONSECUTIVE       = 2

# ---------------------------------------------------------------------------
# Globals for mouse-callback target-lock state
# ---------------------------------------------------------------------------
selected_track_id = None          # track_id of the locked target, or None
current_tracked_detections = None # latest tracked Detections (set each frame)
display_scale = DISPLAY_SCALE     # stored so mouse callback can invert it

# ---------------------------------------------------------------------------
# Colors (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_DEFAULT_BOX   = (0, 200, 0)      # green — unselected tracks
COLOR_LOCKED_BOX    = (0, 0, 255)      # red   — locked target
COLOR_LOST_TEXT     = (0, 0, 255)      # red   — "TARGET LOST" overlay
COLOR_LABEL_BG      = (40, 40, 40)     # dark gray label background
COLOR_LABEL_TEXT    = (255, 255, 255)   # white label text
COLOR_FPS_TEXT      = (0, 255, 255)    # cyan  — FPS counter

THICKNESS_DEFAULT = 2
THICKNESS_LOCKED  = 3
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL  = 0.55
FONT_SCALE_STATUS = 0.7


def print_log(msg):
    print(msg, flush=True)


def on_mouse(event, x, y, flags, param):
    """Mouse callback: left-click inside a tracked box to lock/unlock target.

    Coordinates from the display window are in the upscaled space; we
    convert back to model-space (original 480p) before hit-testing boxes.
    """
    global selected_track_id, current_tracked_detections, display_scale

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Map display-space click → model-space coordinates
    mx = int(x / display_scale)
    my = int(y / display_scale)

    dets = current_tracked_detections
    if dets is None or dets.tracker_id is None or len(dets) == 0:
        if selected_track_id is not None:
            print_log(f"[TARGET] Unlocked (clicked empty space)")
            selected_track_id = None
        return

    # Hit-test in model-space coordinates
    clicked_id = None
    for i in range(len(dets)):
        x1, y1, x2, y2 = dets.xyxy[i].astype(int)
        if x1 <= mx <= x2 and y1 <= my <= y2:
            clicked_id = int(dets.tracker_id[i])
            break

    if clicked_id is not None:
        if selected_track_id == clicked_id:
            return
        selected_track_id = clicked_id
        print_log(f"[TARGET] Locked on track ID {selected_track_id}")
    else:
        if selected_track_id is not None:
            print_log(f"[TARGET] Unlocked (clicked empty space)")
            selected_track_id = None


def draw_label(frame, text, x, y, bg_color, text_color, font_scale=FONT_SCALE_LABEL):
    """Draw a text label with a filled background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
    lx = max(x, 0)
    ly = max(y - th - baseline - 4, 0)
    cv2.rectangle(frame, (lx, ly), (lx + tw + 6, ly + th + baseline + 4), bg_color, -1)
    cv2.putText(frame, text, (lx + 3, ly + th + 2), FONT, font_scale, text_color, 1, cv2.LINE_AA)


def annotate_frame(frame, detections):
    """Draw tracked bounding boxes on the frame with target-lock awareness."""
    global selected_track_id

    annotated = frame.copy()

    if detections is None or detections.tracker_id is None or len(detections) == 0:
        if selected_track_id is not None:
            draw_target_lost_overlay(annotated)
        return annotated

    active_ids = set(detections.tracker_id.astype(int))
    target_visible = selected_track_id is not None and selected_track_id in active_ids

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        tid = int(detections.tracker_id[i])
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0

        is_locked = (selected_track_id is not None and tid == selected_track_id)

        if is_locked:
            box_color = COLOR_LOCKED_BOX
            thickness = THICKNESS_LOCKED
            label = f"LOCKED  ID {tid}  {conf:.2f}"
            label_bg = (0, 0, 180)
        else:
            box_color = COLOR_DEFAULT_BOX
            thickness = THICKNESS_DEFAULT
            label = f"ID {tid}  {conf:.2f}"
            label_bg = COLOR_LABEL_BG

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        draw_label(annotated, label, x1, y1, label_bg, COLOR_LABEL_TEXT)

    if selected_track_id is not None and not target_visible:
        draw_target_lost_overlay(annotated)

    return annotated


def draw_target_lost_overlay(frame):
    """Draw a 'TARGET LOST — searching...' banner in the top-left corner."""
    text = "TARGET LOST - searching..."
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE_STATUS, 2)
    pad = 12
    cv2.rectangle(frame, (0, 0), (tw + pad * 2, th + baseline + pad * 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, th + pad), FONT, FONT_SCALE_STATUS, COLOR_LOST_TEXT, 2, cv2.LINE_AA)


def draw_fps(frame, fps):
    """Draw an FPS counter in the bottom-left corner."""
    h = frame.shape[0]
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (8, h - 10), FONT, 0.5, COLOR_FPS_TEXT, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global selected_track_id, current_tracked_detections, display_scale

    # -----------------------------------------------------------------------
    # [1/3] Load model
    # -----------------------------------------------------------------------
    print_log("\n[2/3] Loading RF-DETR model checkpoint...")
    checkpoint_path = r"e:\Human Detection Dataset\rfdetr_human_detection\checkpoint_best_ema.pth"

    t_ckpt = time.time()
    try:
        if WEBCAM_TEST_MODE:
            print_log("      [WEBCAM TEST MODE] Loading standard COCO model instead of drone checkpoint...")
            model = RFDETRMedium()
        else:
            model = RFDETRMedium.from_checkpoint(checkpoint_path)
        print_log(f"      [OK] Model weights loaded in {time.time() - t_ckpt:.1f} seconds!")
    except Exception as e:
        print_log(f"\nError loading model: {e}")
        return


    # -----------------------------------------------------------------------
    # [3/3] Open webcam / video source
    # -----------------------------------------------------------------------
    print_log("\n[3/3] Searching for an available webcam...")
    cap = None

    # On Windows, CAP_DSHOW on index 0 is fastest and avoids MSMF hangs
    search_order = [
        (0, cv2.CAP_DSHOW, "DirectShow"),
        (0, cv2.CAP_ANY, "Default"),
        (1, cv2.CAP_DSHOW, "DirectShow"),
        (0, cv2.CAP_MSMF, "Media Foundation"),
    ]

    for index, backend_id, name in search_order:
        print_log(f"      Testing camera index {index} ({name})...")
        try:
            temp_cap = cv2.VideoCapture(index, backend_id)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret and frame is not None and frame.size > 0:
                    cap = temp_cap
                    print_log(f"      Success! Connected to camera index {index} using {name}.")
                    break
                else:
                    temp_cap.release()
            else:
                temp_cap.release()
        except Exception:
            pass


    if cap is None or not cap.isOpened():
        print_log("\nError: Could not open any webcam.")
        print_log("Please check:")
        print_log("  1. Is another app (Zoom, Teams, Camera app) currently using your webcam?")
        print_log("  2. Check Windows Settings -> Privacy & Security -> Camera -> Ensure 'Let apps access your camera' is ON.")
        return

    # Set 480p capture — matches drone video link bandwidth
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print_log(f"      Capture resolution: {actual_w}x{actual_h}")

    # -----------------------------------------------------------------------
    # [3/3] Start video feed with tracking + target lock
    # -----------------------------------------------------------------------
    print_log("\n[READY] Webcam video feed starting...")
    print_log("===================================================")
    print_log(f"   Mode     : Drone 480p  (detect @ threshold {DETECTION_THRESHOLD})")
    print_log(f"   Display  : {DISPLAY_SCALE}x upscale for readability")
    print_log("   CONTROLS:")
    print_log("     Left-click a person  -> Lock target")
    print_log("     Left-click empty     -> Unlock target")
    print_log("     'u' key              -> Unlock target")
    print_log("     'q' key / ESC        -> Quit demo")
    print_log("===================================================")

    # --- Tracker setup (Appearance-based ReID) ----------------------------
    tracker = DeepSort(
        max_age=30,            # Keep lost tracks alive for ~1 second (default 30)
        n_init=3,              # Require 3 consecutive frames to confirm a track (filters noise)
        max_iou_distance=0.7,  # Max IOU distance for matching
        embedder='mobilenet'   # Lightweight visual feature extractor
    )

    window_name = "RF-DETR Drone Human Detection - Press 'q' to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    # FPS tracking
    fps = 0.0
    frame_times = []

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret or frame is None:
            print_log("Warning: Failed to grab webcam frame.")
            break

        try:
            # --- Detection ------------------------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb_frame, threshold=DETECTION_THRESHOLD)

            # Filter for humans (class 1 for person in RF-DETR base) if using the base model
            if WEBCAM_TEST_MODE and isinstance(detections, sv.Detections) and len(detections) > 0:
                # In RF-DETR's default mapping, 'person' is class_id 1
                detections = detections[detections.class_id == 1]

            # --- Tracking -------------------------------------------------
            ds_detections = []
            if isinstance(detections, sv.Detections) and len(detections) > 0:
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    conf = detections.confidence[i]
                    w = x2 - x1
                    h = y2 - y1
                    ds_detections.append(([x1, y1, w, h], conf, 0))
            
            tracks = tracker.update_tracks(ds_detections, frame=frame)
            
            # Convert DeepSORT tracks back to supervision Detections for rendering
            tracked_boxes = []
            tracked_conf = []
            tracked_ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                # Do not render predicted tracks (ghosts) if they haven't been matched recently
                if track.time_since_update > 1:
                    continue
                
                track_id = int(track.track_id)
                ltrb = track.to_ltrb() # [left, top, right, bottom]
                tracked_boxes.append(ltrb)
                tracked_conf.append(track.det_conf if track.det_conf else 0.99)
                tracked_ids.append(track_id)
            
            if len(tracked_boxes) > 0:
                tracked = sv.Detections(
                    xyxy=np.array(tracked_boxes),
                    confidence=np.array(tracked_conf),
                    class_id=np.zeros(len(tracked_boxes), dtype=int),
                    tracker_id=np.array(tracked_ids)
                )
            else:
                tracked = sv.Detections.empty()

            current_tracked_detections = tracked

            # --- Render (on original 480p frame) --------------------------
            annotated_frame = annotate_frame(frame, tracked)

            # --- FPS counter on annotated frame ---------------------------
            draw_fps(annotated_frame, fps)

        except Exception as e:
            print_log(f"Inference error: {e}")
            annotated_frame = frame

        # --- Display: upscale for operator readability --------------------
        if DISPLAY_SCALE != 1.0:
            disp_w = int(annotated_frame.shape[1] * DISPLAY_SCALE)
            disp_h = int(annotated_frame.shape[0] * DISPLAY_SCALE)
            display_frame = cv2.resize(annotated_frame, (disp_w, disp_h),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = annotated_frame

        cv2.imshow(window_name, display_frame)

        # --- FPS calculation (rolling average of last 30 frames) ----------
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

        # --- Key handling -------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print_log("\nExiting demo...")
            break
        elif key == ord('u'):
            if selected_track_id is not None:
                print_log(f"[TARGET] Unlocked via 'u' key (was ID {selected_track_id})")
                selected_track_id = None

    cap.release()
    cv2.destroyAllWindows()
    print_log("Demo closed.")


if __name__ == "__main__":
    main()
