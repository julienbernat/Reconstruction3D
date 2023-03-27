import mediapipe as mp
import cv2 as cv

# https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
rightEyeIDLandmarks = [398, 384, 385, 386, 387, 388, 466, 763, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeIDLandmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

rightEyePixelLEFT = []
leftEyePixelLEFT = []

rightEyePixelRIGHT = []
leftEyePixelRIGHT = []

def eyesDetection(img):
    
    imgL = img[:, :int(len(img[0])/2)]
    imgR = img[:, int(len(img[0])/2):]

    # LEFT IMAGE
    ih, iw, ic = imgL.shape
    print(imgL.shape)
    faceMeshSolution = mp.solutions.face_mesh
    faceMesh = faceMeshSolution.FaceMesh(max_num_faces=1) # 2 faces because of 2 cameras..
    results = faceMesh.process(imgL)

    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            for id, landmark in enumerate(faceLandmarks.landmark):

                if id in rightEyeIDLandmarks:
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    rightEyePixelLEFT.append((x, y))
                    cv.circle(imgL, (x, y), radius=1, color=(255, 255, 255), thickness=3)


                if id in leftEyeIDLandmarks:
                    x, y= int(landmark.x * iw), int(landmark.y * ih)
                    leftEyePixelLEFT.append((x, y))
                    cv.circle(imgL, (x, y), radius=1, color=(255, 255, 255), thickness=3)

    # LEFT IMAGE
    ih, iw, ic = imgR.shape
    faceMeshSolution = mp.solutions.face_mesh
    faceMesh = faceMeshSolution.FaceMesh(max_num_faces=1) # 2 faces because of 2 cameras..
    results = faceMesh.process(imgR)

    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            for id, landmark in enumerate(faceLandmarks.landmark):

                if id in rightEyeIDLandmarks:
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    rightEyePixelRIGHT.append((x, y))
                    cv.circle(imgR, (x, y), radius=1, color=(255, 255, 255), thickness=3)


                if id in leftEyeIDLandmarks:
                    x, y= int(landmark.x * iw), int(landmark.y * ih)
                    leftEyePixelRIGHT.append((x, y))
                    cv.circle(imgR, (x, y), radius=1, color=(255, 255, 255), thickness=3)

    return leftEyePixelLEFT, rightEyePixelLEFT, leftEyePixelRIGHT, rightEyePixelRIGHT