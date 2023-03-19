import cv2 as cv
from Depth import CalculateDisparity

def processFrame(frame,intrinsicL, intrinsicR, distL, distR):
    depthMap = CalculateDisparity(frame,intrinsicL, intrinsicR, distL, distR)
    cv.imshow("frame", depthMap)
  

def realTimeCapture(intrinsicL, intrinsicR, distL, distR):
  vid = cv.VideoCapture(0)
  while (True):
      ret, frame = vid.read()
      processFrame(frame,intrinsicL, intrinsicR, distL, distR)
      if cv.waitKey(1) & 0xFF == ord('q'):
          break
  vid.release()
  cv.destroyAllWindows()
