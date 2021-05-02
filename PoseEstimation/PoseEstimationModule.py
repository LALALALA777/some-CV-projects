import mediapipe as mp
import cv2
import time


class PoseEstimation():
    def __init__(self, static_image_mode=False,
                 upper_body_only=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self)
        self.drawSpace = self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2)

    def find_pose(self, img, draw=True):
        results = self.pose.process(img)
        mfls = results.pose_landmarks
        coordinates = []
        if mfls is not None:
            if draw:
                self.mpDraw.draw_landmarks(img, mfls, self.mpPose.POSE_CONNECTIONS,
                                           self.drawSpace, self.drawSpace)

            for id, lm in enumerate(mfls.landmark):
                ih, iw, ic = img.shape
                x, y = ih * lm.x, iw * lm.y
                coordinates.append((id, x, y))

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), coordinates


def main():
    #cap = cv2.VideoCapture("videos/1.mp4")
    cap = cv2.VideoCapture(0)
    detector = PoseEstimation()
    ptime = 0

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, coordinates = detector.find_pose(imgRGB, draw=True)
        print(coordinates)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'fps:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
        cv2.namedWindow('Pose')
        cv2.imshow('Pose', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
