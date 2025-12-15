import cv2
import time
import mediapipe as mp

# Ziel-FPS für loop (ohne sleep(), nur Frame-Skipping)
TARGET_FPS = 30
last_frame_time = 0

# MediaPipe Task-APIs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_gesture(frame):
    base_options = python.BaseOptions(model_asset_path="C:/Users/ulisc/Workspace/rysa/gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )

    recognizer = vision.GestureRecognizer.create_from_options(options)
    # RGB-Bild für MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


    # Gesture Recognition
    result = recognizer.recognize(mp_image)

    if result.gestures:
        gesture = result.gestures[0][0].category_name
        if gesture == "Thumb_Up":
            print("Daumen hoch", end="\r")
        else:
            print(gesture, end="\r")

    cv2.imshow("Gesture Recognizer", frame)

    
def main():
    global last_frame_time

    # Optionen + Modelldatei
    base_options = python.BaseOptions(model_asset_path="C:/Users/ulisc/Workspace/rysa/gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)

    while True:
        now = time.time()
        if now - last_frame_time < 1.0 / TARGET_FPS:
            continue
        last_frame_time = now

        ret, frame = cap.read()
        if not ret:
            continue

        # RGB-Bild für MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in ms
        timestamp_ms = int(now * 1000)

        # Gesture Recognition
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.gestures:
            gesture = result.gestures[0][0].category_name
            if gesture == "Thumb_Up":
                print("Daumen hoch", end="\r")
            else:
                print(gesture, end="\r")

        cv2.imshow("Gesture Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
