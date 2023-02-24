import numpy as np
import cv2 as cv
import glob

cbSize = (5, 7)
imgSize = [1314, 1027]
stereoChessboards = glob.glob("./StereoChessboards/*")

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objP = np.zeros((cbSize[0]*cbSize[1], 3), np.float32)

objP[:, :2] = np.mgrid[0:cbSize[0], 0:cbSize[1]].T.reshape(-1, 2)*20

objPts = []
imgPtsL = []
imgPtsR = []

def CalibrateStereo():

    for chessBoard in stereoChessboards:
        img = cv.imread(chessBoard)
        imgL = img[:, :int(len(img[0])/2)]
        imgR = img[:, int(len(img[0])/2):]

        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        foundL, cornersL = cv.findChessboardCorners(grayL, cbSize, None)

        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        foundR, cornersR = cv.findChessboardCorners(grayR, cbSize, None)

        if foundL and foundR== True:

            objPts.append(objP)

            cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgPtsL.append(cornersL)
            
            cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgPtsR.append(cornersR)

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objPts, imgPtsL, imgSize, None, None)
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, (imgSize[0], imgSize[1]), 1, (imgSize[0], imgSize[1]))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objPts, imgPtsR, imgSize, None, None)
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, (imgSize[0], imgSize[1]), 1, (imgSize[0], imgSize[1]))

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objPts, imgPtsL, imgPtsR, newCameraMatrixL, distL, newCameraMatrixR, distR, (imgSize[0], imgSize[1]), criteria, flags)

    print("retval:\n", retStereo)
    print("\nRight camera intrinsic:\n", newCameraMatrixR)
    print("\nLeft camera intrinsic:\n", newCameraMatrixL)
    print("\nFundamental:\n", fundamentalMatrix)
    print("\nEssential:\n", essentialMatrix)
    print("\nTranslation:\n", trans)
    print("\nRotation:\n", rot)
    print("\nDistorsion left:\n", distL)
    print("\nDistorsion right:\n", distR)

    return fundamentalMatrix, newCameraMatrixL

if __name__ == "__main__":
  F, m = CalibrateStereo()
  print("\nFundamental matrix:\n", F)
  print("\nLeft camera intrinsic matrix:\n", m)
