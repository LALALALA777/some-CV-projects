import mediapipe as mp
import cv2
import time

#cap = cv2.VideoCapture("videos/1.mp4")
cap = cv2.VideoCapture(0)   # driving camera
ptime = 0

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()    # class
#drawSpace = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    pose_landmarks = results.pose_landmarks
    #print(results.pose_landmarks)
    if pose_landmarks is not None:
        mpDraw.draw_landmarks(img, pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(pose_landmarks.landmark):
            ih, iw, ic = img.shape
            x, y = int(iw*lm.x), int(ih*lm.y)
            cv2.circle(img, (x, y), 10, (255, 0, 255,), cv2.FILLED)


    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'fps:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("frame", img)
    cv2.waitKey(1)