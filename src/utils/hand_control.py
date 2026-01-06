import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandControl:
    def __init__(self):
        # -----------------------------
        # Configuration
        # -----------------------------  
        self.MODEL_PATH = "C:/Users/ulisc/Workspace/rysa/gesture_recognizer.task"

        # -----------------------------
        # MediaPipe Setup (IMAGE Mode)
        # -----------------------------
        self.base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.IMAGE,
        )

        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def recognize_gesture(self,frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.recognizer.recognize(mp_image)
        gesture = None
        if result.gestures:
            gesture = result.gestures[0][0].category_name

        return gesture
    




if __name__ == "__main__":
    # -----------------------------
    # Camera
    # -----------------------------
    FRAME_STEP = 5 
    cap = cv2.VideoCapture(0)
    frame_index = 0

    # Recognizer
    Recognizer = HandControl()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Only evlaluate every n-th frame
        elif frame_index % FRAME_STEP == 0:
            gesture = Recognizer.recognize_gesture(frame)
            print(f"Geste: {gesture}", end="\r")

        frame_index += 1



        cv2.imshow("Gesture Recognizer (IMAGE Mode)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
