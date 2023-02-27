import numpy as np
import cv2 as cv
import glob

stereoChessboards = glob.glob("./StereoChessboards/*")

#3D points of chessboard
objPts = []

#2D points of left and right image
imgPtsL = []
imgPtsR = []

#Chssboard size
cbSize = (5, 7)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Image size
imgSize = [1314, 1027]

#World coordinates for 3D points
coord = np.zeros((cbSize[0]*cbSize[1], 3), np.float32)
coord[:, :2] = np.mgrid[0:cbSize[0], 0:cbSize[1]].T.reshape(-1, 2)

def getIntrinsicMatrix(imgPts):
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objPts, imgPts, imgSize, None, None)
    optimalCameraMatrix, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (imgSize[0], imgSize[1]), 1, (imgSize[0], imgSize[1]))
    return  optimalCameraMatrix , distCoeffs


def CalibrateCamera():
    for chessBoard in stereoChessboards:
        img = cv.imread(chessBoard)

        #Separate left and right image
        imgL = img[:, :int(len(img[0])/2)]
        imgR = img[:, int(len(img[0])/2):]

        #Find corners of the chessboard
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        foundL, cornersL = cv.findChessboardCorners(grayL, cbSize, None)

        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        foundR, cornersR = cv.findChessboardCorners(grayR, cbSize, None)

        if foundL and foundR== True:
            objPts.append(coord)
            
            #Add 2d image points of the chessboard corners
            cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgPtsL.append(cornersL)
            
            cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgPtsR.append(cornersR)

    #Calculate intrisic matrices
    intrinsicL, distL = getIntrinsicMatrix(imgPtsL)
    intrinsicR, distR = getIntrinsicMatrix(imgPtsR)

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    #Calculate fundamental and essential matrix with stereo calibration
    retStereo, intrinsicL, distL, intrinsicR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objPts, imgPtsL, imgPtsR, intrinsicL, distL, intrinsicR, distR, (imgSize[0], imgSize[1]), criteria, flags)

    print("retval:\n", retStereo)
    print("\nRight camera intrinsic matrix:\n", intrinsicR)
    print("\nLeft camera intrinsic matrix:\n", intrinsicL)
    print("\nFundamental:\n", fundamentalMatrix)
    print("\nEssential:\n", essentialMatrix)
    print("\nTranslation:\n", trans)
    print("\nRotation:\n", rot)
    print("\nDistorsion left:\n", distL)
    print("\nDistorsion right:\n", distR)

    return fundamentalMatrix, intrinsicL
