import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
from enum import Enum


class GestureType(Enum):
    NONE = "None"
    CURSOR_MOVE = "Cursor Move"
    LEFT_CLICK = "Left Click"
    RIGHT_CLICK = "Right Click"
    SCROLL_UP = "Scroll Up"
    SCROLL_DOWN = "Scroll Down"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"


class HandGestureController:

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        )

        self.screen_width, self.screen_height = pyautogui.size()
        self.cam_width, self.cam_height = 640, 480

        self.frame_reduction = 100

        self.smoothing_frames = 5
        self.prev_positions = deque(maxlen=self.smoothing_frames)

        self.smoothing_factor = 0.5
        self.prev_cursor_x = 0
        self.prev_cursor_y = 0

        # FIXED: Separate detection for different gestures
        self.pinch_threshold = 0.05  # For clicks
        self.scroll_threshold = 0.03
        
        # ZOOM: Use THREE fingers (thumb + index + middle) to avoid conflict
        self.zoom_mode_active = False
        self.zoom_base_distance = None
        self.zoom_trigger_threshold = 0.15  # Significant spread/contract needed
        
        # Mode tracking
        self.current_mode = "CURSOR"

        self.gesture_cooldown = 0.3
        self.last_click_time = 0
        self.last_scroll_time = 0
        self.last_zoom_time = 0

        self.left_click_locked = False
        self.right_click_locked = False
        self.zoom_locked = False

        self.prev_scroll_y = None
        self.scroll_sensitivity = 20

        self.prev_time = 0
        self.fps = 0

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def normalize_distance(self, distance, hand_size):
        return distance / hand_size if hand_size > 0 else 0

    def get_hand_size(self, landmarks):
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        return self.calculate_distance(
            (wrist.x, wrist.y),
            (middle_tip.x, middle_tip.y)
        )

    def apply_smoothing(self, raw_x, raw_y):
        smooth_x = (self.smoothing_factor * raw_x + 
                   (1 - self.smoothing_factor) * self.prev_cursor_x)
        smooth_y = (self.smoothing_factor * raw_y + 
                   (1 - self.smoothing_factor) * self.prev_cursor_y)

        self.prev_cursor_x = smooth_x
        self.prev_cursor_y = smooth_y

        return int(smooth_x), int(smooth_y)
    
    def is_fist_closed(self, landmarks):
        """Check if hand is in fist position (all fingers down)"""
        # Check if all fingertips are below their base joints
        fingers_down = 0
        
        # Index finger
        if landmarks[8].y > landmarks[6].y:
            fingers_down += 1
        # Middle finger
        if landmarks[12].y > landmarks[10].y:
            fingers_down += 1
        # Ring finger
        if landmarks[16].y > landmarks[14].y:
            fingers_down += 1
        # Pinky
        if landmarks[20].y > landmarks[18].y:
            fingers_down += 1
            
        return fingers_down >= 3  # At least 3 fingers down = fist

    def detect_gesture(self, landmarks):
        """
        COMPLETELY REDESIGNED gesture detection:
        - CURSOR: Index finger extended (default)
        - CLICK: Thumb touches specific finger
        - SCROLL: Two fingers extended, vertical movement
        - ZOOM: Make FIST, then open hand = Zoom In OR close hand = Zoom Out
        """
        current_time = time.time()
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]

        hand_size = self.get_hand_size(landmarks)

        # Calculate key distances
        thumb_index_dist = self.calculate_distance(
            (thumb_tip.x, thumb_tip.y),
            (index_tip.x, index_tip.y)
        )
        norm_thumb_index = self.normalize_distance(thumb_index_dist, hand_size)

        index_middle_dist = self.calculate_distance(
            (index_tip.x, index_tip.y),
            (middle_tip.x, middle_tip.y)
        )
        norm_index_middle = self.normalize_distance(index_middle_dist, hand_size)

        # Check if fist
        is_fist = self.is_fist_closed(landmarks)

        # ===== ZOOM DETECTION (REDESIGNED) =====
        # ZOOM MODE: Make a fist to enter zoom mode
        # Then: Open hand = Zoom In, Close hand = Zoom Out
        
        if is_fist and not self.zoom_mode_active:
            # Enter zoom mode with fist
            self.zoom_mode_active = True
            # Calculate average distance of all fingertips from palm center
            palm_center_x = (landmarks[0].x + landmarks[9].x) / 2
            palm_center_y = (landmarks[0].y + landmarks[9].y) / 2
            
            avg_distance = (
                self.calculate_distance((landmarks[8].x, landmarks[8].y), (palm_center_x, palm_center_y)) +
                self.calculate_distance((landmarks[12].x, landmarks[12].y), (palm_center_x, palm_center_y)) +
                self.calculate_distance((landmarks[16].x, landmarks[16].y), (palm_center_x, palm_center_y)) +
                self.calculate_distance((landmarks[20].x, landmarks[20].y), (palm_center_x, palm_center_y))
            ) / 4
            
            self.zoom_base_distance = avg_distance
            self.current_mode = "ZOOM"
            return GestureType.NONE
        
        elif self.zoom_mode_active:
            # In zoom mode - track hand opening/closing
            palm_center_x = (landmarks[0].x + landmarks[9].x) / 2
            palm_center_y = (landmarks[0].y + landmarks[9].y) / 2
            
            current_avg_distance = (
                self.calculate_distance((landmarks[8].x, landmarks[8].y), (palm_center_x, palm_center_y)) +
                self.calculate_distance((landmarks[12].x, landmarks[12].y), (palm_center_x, palm_center_y)) +
                self.calculate_distance((landmarks[16].x, landmarks[16].y), (palm_center_x, palm_center_y)) +
                self.calculate_distance((landmarks[20].x, landmarks[20].y), (palm_center_x, palm_center_y))
            ) / 4
            
            if self.zoom_base_distance is not None:
                distance_change = current_avg_distance - self.zoom_base_distance
                norm_change = self.normalize_distance(abs(distance_change), hand_size)
                
                if norm_change > self.zoom_trigger_threshold:
                    if not self.zoom_locked and (current_time - self.last_zoom_time) > self.gesture_cooldown:
                        self.zoom_base_distance = current_avg_distance
                        
                        if distance_change > 0:
                            # Hand opening = Zoom In
                            return GestureType.ZOOM_IN
                        else:
                            # Hand closing = Zoom Out
                            return GestureType.ZOOM_OUT
            
            # Exit zoom mode if hand fully open (all fingers extended)
            fingers_extended = (
                landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                landmarks[16].y < landmarks[14].y and
                landmarks[20].y < landmarks[18].y
            )
            
            if fingers_extended:
                self.zoom_mode_active = False
                self.zoom_base_distance = None
                self.current_mode = "CURSOR"
            
            return GestureType.NONE

        # ===== CLICK DETECTION =====
        # Left click: Thumb and index very close
        if norm_thumb_index < self.pinch_threshold:
            if not self.left_click_locked and (current_time - self.last_click_time) > self.gesture_cooldown:
                self.current_mode = "CLICK"
                return GestureType.LEFT_CLICK
            return GestureType.NONE
        else:
            self.left_click_locked = False

        # Right click: Index and middle pinch
        if norm_index_middle < self.pinch_threshold:
            if not self.right_click_locked and (current_time - self.last_click_time) > self.gesture_cooldown:
                self.current_mode = "CLICK"
                return GestureType.RIGHT_CLICK
            return GestureType.NONE
        else:
            self.right_click_locked = False

        # ===== SCROLL DETECTION =====
        fingers_extended = (
            index_tip.y < index_mcp.y and 
            middle_tip.y < middle_mcp.y
        )

        if fingers_extended:
            avg_finger_y = (index_tip.y + middle_tip.y) / 2

            if self.prev_scroll_y is not None:
                scroll_delta = (self.prev_scroll_y - avg_finger_y) * self.cam_height

                if abs(scroll_delta) > self.scroll_sensitivity:
                    if (current_time - self.last_scroll_time) > 0.1:
                        self.prev_scroll_y = avg_finger_y
                        self.current_mode = "SCROLL"

                        if scroll_delta > 0:
                            return GestureType.SCROLL_UP
                        else:
                            return GestureType.SCROLL_DOWN
            
            self.prev_scroll_y = avg_finger_y
        else:
            self.prev_scroll_y = None

        # ===== DEFAULT: CURSOR MOVEMENT =====
        self.current_mode = "CURSOR"
        return GestureType.CURSOR_MOVE

    def execute_action(self, gesture, landmarks):
        current_time = time.time()

        if gesture == GestureType.CURSOR_MOVE:
            index_tip = landmarks[8]

            x = np.interp(
                index_tip.x,
                [self.frame_reduction / self.cam_width, 
                 1 - self.frame_reduction / self.cam_width],
                [0, self.screen_width]
            )
            y = np.interp(
                index_tip.y,
                [self.frame_reduction / self.cam_height, 
                 1 - self.frame_reduction / self.cam_height],
                [0, self.screen_height]
            )

            smooth_x, smooth_y = self.apply_smoothing(x, y)

            safe_x = max(1, min(smooth_x, self.screen_width - 1))
            safe_y = max(1, min(smooth_y, self.screen_height - 1))
            pyautogui.moveTo(safe_x, safe_y, duration=0)

        elif gesture == GestureType.LEFT_CLICK:
            if not self.left_click_locked:
                pyautogui.click()
                self.last_click_time = current_time
                self.left_click_locked = True

        elif gesture == GestureType.RIGHT_CLICK:
            if not self.right_click_locked:
                pyautogui.rightClick()
                self.last_click_time = current_time
                self.right_click_locked = True

        elif gesture == GestureType.SCROLL_UP:
            pyautogui.scroll(1)
            self.last_scroll_time = current_time

        elif gesture == GestureType.SCROLL_DOWN:
            pyautogui.scroll(-1)
            self.last_scroll_time = current_time

        elif gesture == GestureType.ZOOM_IN:
            if not self.zoom_locked:
                pyautogui.hotkey('ctrl', '+')
                self.last_zoom_time = current_time
                self.zoom_locked = True

        elif gesture == GestureType.ZOOM_OUT:
            if not self.zoom_locked:
                pyautogui.hotkey('ctrl', '-')
                self.last_zoom_time = current_time
                self.zoom_locked = True

        if gesture not in [GestureType.ZOOM_IN, GestureType.ZOOM_OUT]:
            self.zoom_locked = False

    def draw_info(self, frame, gesture, fps):
        """Draw info with clearer mode indicator"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Color based on mode
        mode_colors = {
            "CURSOR": (0, 255, 0),    # Green
            "ZOOM": (0, 165, 255),    # Orange
            "CLICK": (255, 0, 255),   # Magenta
            "SCROLL": (255, 255, 0)   # Yellow
        }
        mode_color = mode_colors.get(self.current_mode, (255, 255, 255))

        cv2.putText(frame, f"FPS: {fps}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: {self.current_mode}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        cv2.putText(frame, f"Gesture: {gesture.value}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show zoom status
        if self.zoom_mode_active:
            cv2.putText(frame, "ZOOM READY - Open/Close Hand", (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit", (20, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            print("Please check:")
            print("1. Webcam is connected")
            print("2. Camera permissions are granted")
            print("3. No other application is using the camera")
            return

        print("=" * 70)
        print("     HAND GESTURE MOUSE CONTROL - FINAL VERSION")
        print("=" * 70)
        print("\n🎮 COMPLETE GESTURE GUIDE:")
        print("-" * 70)
        print("\n1️⃣  CURSOR MOVEMENT (Default Mode)")
        print("   👆 Point with index finger")
        print("   → Cursor follows your finger smoothly")
        print()
        print("2️⃣  LEFT CLICK")
        print("   🤏 Pinch: Bring thumb and index finger together")
        print("   → Quick pinch and release")
        print()
        print("3️⃣  RIGHT CLICK")
        print("   ✌️🤏 Pinch: Bring index and middle fingers together")
        print("   → Quick pinch and release")
        print()
        print("4️⃣  SCROLL UP/DOWN")
        print("   ✌️ Extend index and middle fingers (peace sign)")
        print("   ⬆️  Move hand UP = Scroll Up")
        print("   ⬇️  Move hand DOWN = Scroll Down")
        print()
        print("5️⃣  ZOOM IN/OUT (3-Step Process)")
        print("   Step 1: ✊ Make a FIST (close all fingers)")
        print("           → Screen shows 'Mode: ZOOM' (Orange)")
        print("           → Message: 'ZOOM READY'")
        print()
        print("   Step 2: Perform zoom action:")
        print("           🖐️  OPEN hand (spread fingers) = ZOOM IN")
        print("           ✊  CLOSE hand (tighten fist) = ZOOM OUT")
        print()
        print("   Step 3: 🖐️  Fully extend all fingers to EXIT zoom mode")
        print("           → Returns to cursor control")
        print()
        print("=" * 70)
        print("\n💡 TIPS:")
        print("  • Keep hand 1-2 feet from camera")
        print("  • Good lighting helps detection")
        print("  • Make gestures deliberately")
        print("  • Watch the 'Mode' indicator on screen")
        print("  • For zoom: FIST first, then open/close")
        print("\n🎨 MODE COLORS:")
        print("  • Green  = CURSOR (moving mouse)")
        print("  • Orange = ZOOM (zoom active)")
        print("  • Yellow = SCROLL (scrolling)")
        print("  • Magenta = CLICK (click detected)")
        print()
        print("Press 'q' in the video window to quit")
        print("=" * 70)

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    print("Warning: Failed to read frame from webcam")
                    continue

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.hands.process(rgb_frame)

                current_gesture = GestureType.NONE

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )

                        landmarks = hand_landmarks.landmark

                        current_gesture = self.detect_gesture(landmarks)

                        self.execute_action(current_gesture, landmarks)

                        h, w, _ = frame.shape
                        x_coords = [lm.x * w for lm in landmarks]
                        y_coords = [lm.y * h for lm in landmarks]
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                else:
                    # Reset when no hand detected
                    self.current_mode = "CURSOR"
                    self.zoom_mode_active = False

                current_time = time.time()
                if self.prev_time != 0:
                    self.fps = int(1 / (current_time - self.prev_time))
                self.prev_time = current_time

                self.draw_info(frame, current_gesture, self.fps)

                cv2.imshow('Hand Gesture Mouse Control - FINAL', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nShutting down...")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        except Exception as e:
            print(f"\nError occurred: {str(e)}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("System stopped successfully")


def main():
    try:
        controller = HandGestureController()
        controller.run()

    except Exception as e:
        print(f"Fatal Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check webcam permissions")
        print("3. Close other applications using the camera")


if __name__ == "__main__":
    main()