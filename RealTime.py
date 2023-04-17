from Calibration import CalibrateCamera
from Matching import Matching
from Depth import CalculateDepth
from imagePreparation import ImagePreparation
from EyesDetection import eyesDetection
import cv2 as cv


# https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
def hconcat_resize(img_list, 
                   interpolation 
                   = cv.INTER_CUBIC):
      # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
      
    # image resizing 
    im_list_resize = [cv.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation
                                 = interpolation) 
                      for img in img_list]
      
    # return final image
    return cv.hconcat(im_list_resize)

def processFrame(frame,intrinsicL, intrinsicR, distL, distR, cameraMatrixL, cameraMatrixR, processing_unit):
    width = 1028
    height = 512
    dim = (width, height)
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    frame_eyes = frame.copy()
    leftEyePixelsLEFT, rightEyePixelsLEFT, leftEyePixelsRIGHT, rightEyePixelsRIGHT = eyesDetection(frame_eyes)
 

    depth, imgWithDistance = CalculateDepth(frame_eyes, intrinsicL, cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR, leftEyePixelsLEFT, rightEyePixelsLEFT, processing_unit)
    
    # # il ne considère pas la variable disparity comme une image, il faut aller
    # # chercher l'image avec cv.imread et ca ca retourne une image 1 channel....
    imgDisparity = cv.imread("./result/disparity.jpg")
    newImage = hconcat_resize([imgWithDistance, imgDisparity])
    cv.imshow("frame", newImage)
  

def realTimeCapture(intrinsicL, intrinsicR, distL, distR, cameraMatrixL, cameraMatrixR, processing_unit):
  vid = cv.VideoCapture(0)
  while (True):
      ret, frame = vid.read()
      if ret == True:
        
        #cv.imshow( "HEllo ",frame)
        #frame = cv.imread("./StereoImages/Center38cm.png", 3)
        processFrame(frame, intrinsicL, intrinsicR, distL, distR, cameraMatrixL, cameraMatrixR, processing_unit)
      if cv.waitKey(1) & 0xFF == ord('q'):
          break
  vid.release()
  cv.destroyAllWindows()

if __name__ == "__main__":
    fundamentalMatrix, intrinsicL,cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR = CalibrateCamera()



    realTimeCapture(intrinsicL, intrinsicR, distL, distR)
