import cv2

class CameraHandler:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return "Error"
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def release(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
