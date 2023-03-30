from Calibration import CalibrateCamera
from Matching import Matching
from Depth import CalculateDepth
from imagePreparation import ImagePreparation
from EyesDetection import eyesDetection
import cv2 as cv

def processFrame(frame,intrinsicL, intrinsicR, distL, distR):
    leftEyePixelsLEFT, rightEyePixelsLEFT, leftEyePixelsRIGHT, rightEyePixelsRIGHT = eyesDetection(frame)

    depth = CalculateDepth(frame, intrinsicL, cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR, leftEyePixelsLEFT, rightEyePixelsLEFT)    
    cv.imshow("frame", depth)
  

def realTimeCapture(intrinsicL, intrinsicR, distL, distR):
  vid = cv.VideoCapture(0)
  while (True):
      ret, frame = vid.read()
      processFrame(frame,intrinsicL, intrinsicR, distL, distR)
      if cv.waitKey(1) & 0xFF == ord('q'):
          break
  vid.release()
  cv.destroyAllWindows()

if __name__ == "__main__":
    fundamentalMatrix, intrinsicL,cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR = CalibrateCamera()



    realTimeCapture(intrinsicL, intrinsicR, distL, distR)
