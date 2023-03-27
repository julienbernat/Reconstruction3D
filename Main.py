from Calibration import CalibrateCamera
from Matching import Matching
from Depth import CalculateDisparity
from RealTime import realTimeCapture
from imagePreparation import ImagePreparation
from EyesDetection import eyesDetection
import cv2 as cv


import argparse

parser = argparse.ArgumentParser(description='Reconstruction 3D.')

parser.add_argument("-f", "--file_number",
                    help="Prints the supplied argument.", type=int)

if __name__ == "__main__":
    fundamentalMatrix, intrinsicL,cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR = CalibrateCamera()
    imgG, imgD, matches, = Matching(
        fundamentalMatrix, parser.parse_args().file_number)

    imgName = "./StereoImages/Center38cm.png"
    img = cv.imread(imgName, 3)

    # cv.imwrite("./result/original.jpg", img)
    cropped = ImagePreparation(img)
    img_landmarks = cropped
    leftEyePixelsLEFT, rightEyePixelsLEFT, leftEyePixelsRIGHT, rightEyePixelsRIGHT = eyesDetection(img_landmarks)
    CalculateDisparity(cropped, intrinsicL, cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR, leftEyePixelsLEFT, rightEyePixelsLEFT)
    
    # realTimeCapture(intrinsicL, intrinksicR, distL, distR)


