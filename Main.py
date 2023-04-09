from Calibration import CalibrateCamera
from Matching import Matching
from Depth import CalculateDepth
from RealTime import realTimeCapture
from imagePreparation import ImagePreparation
from EyesDetection import eyesDetection
import cv2 as cv
import time

import argparse

parser = argparse.ArgumentParser(description='Reconstruction 3D.')

parser.add_argument("-f", "--file_number",
                    help="Prints the supplied argument.", type=int)

if __name__ == "__main__":
    beg = time.time()
    st = time.time()
    fundamentalMatrix, intrinsicL,cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR = CalibrateCamera()
    et = time.time()
    print("Time to calibrate Camera in seconds ", et - st)
    # imgG, imgD, matches, = Matching(fundamentalMatrix, parser.parse_args().file_number)
    imgName = "./StereoImages/Center38cm.png"
    img = cv.imread(imgName, 3)
    eyesImg = cv.imread(imgName, 3)

    st = time.time()
    cropped = ImagePreparation(img)
    croppedEyes = ImagePreparation(eyesImg)
    
    et = time.time()
    print("Time for Image Preparation in secondes : ", et - st)

    st = time.time()
    leftEyePixelsLEFT, rightEyePixelsLEFT, leftEyePixelsRIGHT, rightEyePixelsRIGHT = eyesDetection(croppedEyes)
    et = time.time()
    print("Time for eyes dectection for left and right images in secondes : ", et - st)
    CalculateDepth(cropped, intrinsicL, cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR, leftEyePixelsLEFT, rightEyePixelsLEFT)
    end = time.time()

    print("Total execution time of program in seconds ", end - beg)