import mediapipe as mp
import cv2
import time


class FaceMeshDetector():
    def __init__(self, static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawSpace = self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):
        faces = []
        results = self.faceMesh.process(img)
        mfls = results.multi_face_landmarks
        if mfls:
            for facelms in mfls:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACE_CONNECTIONS,
                                               self.drawSpace, self.drawSpace)

                face = []
                for id, lm in enumerate(facelms.landmark):
                    ih, iw, ic = img.shape
                    x, y = ih * lm.x, iw * lm.y
                    face.append((x, y))

                faces.append(face)

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), faces

def main():
    cap = cv2.VideoCapture("videos/1.mp4")
    #cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(max_num_faces=3)
    ptime = 0
    last_face_num = 0

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, faces = detector.find_face_mesh(imgRGB, draw=True)
        face_num = len(faces)
        if face_num and last_face_num != face_num:
            print(f"Detected face gross is now {face_num}")
            last_face_num = face_num

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'fps:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
        cv2.namedWindow('Faces')
        cv2.imshow('Faces', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()