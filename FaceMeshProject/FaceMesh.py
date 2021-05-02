import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture("videos/1.mp4")
#cap = cv2.VideoCapture(0)   # realtime
ptime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()    # class
drawSpace = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # feceMesh类只接收rgb图像

    results = faceMesh.process(imgRGB)  # process会降低帧数，与图片复杂度正比
    mfls = results.multi_face_landmarks
    if mfls:
        for facelms in mfls:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACE_CONNECTIONS, drawSpace, drawSpace)

        for id, lm in enumerate(facelms.landmark):
            ih, iw, ic = img.shape
            x, y = ih*lm.x, iw*lm.y
            print(id, x, y)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'fps:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("frame", img)
    cv2.waitKey(1)