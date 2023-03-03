import numpy as np
import cv2 as cv
import glob

stereoChessboards = glob.glob("./StereoChessboards/*")

# Chessboard size
cbSize = (5, 6)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)

# Image size
imgSize = [1314, 1027]

# 3D points of chessboard
objPts = []


def CalibrateCamera():
    # World coordinates for 3D points
    coord = np.zeros((cbSize[0]*cbSize[1], 3), np.float32)
    coord[:, :2] = np.mgrid[0:cbSize[0], 0:cbSize[1]].T.reshape(-1, 2)

    # 2D points of left and right image
    imgPtsL = []
    imgPtsR = []

    for chessBoard in stereoChessboards:
        img = cv.imread(chessBoard)

        # Separate left and right image
        imgL = img[:, :int(len(img[0])/2)]
        imgR = img[:, int(len(img[0])/2):]

        # Find corners of the chessboard
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        retL, cornersL = cv.findChessboardCorners(grayL, cbSize, None)

        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        retR, cornersR = cv.findChessboardCorners(grayR, cbSize, None)

        if retL and retR == True:
            objPts.append(coord)

            # Add 2d image points of the chessboard corners
            cv.cornerSubPix(grayL, cornersL, (2, 2), (-1, -1), criteria)
            cv.cornerSubPix(grayR, cornersR, (2, 2), (-1, -1), criteria)

            imgPtsL.append(cornersL)
            imgPtsR.append(cornersR)

            cv.drawChessboardCorners(imgL, cbSize, cornersL, retL)
            cv.drawChessboardCorners(imgR, cbSize, cornersR, retR)
    cv.imwrite("./result/ChessboardCorners.jpg", cv.hconcat([imgL, imgR]))

    heightR, widthR, channelsR = imgR.shape
    retval, cameraMatrixR, distR, rvecs, tvecs = cv.calibrateCamera(
        objPts, imgPtsR, tuple(imgSize), None, None)
    intrinsicR, roi_R = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    # Calculate intrisic  matrices
    heightL, widthL, channelsL = imgL.shape
    retval, cameraMatrixL, distL, rvecs, tvecs = cv.calibrateCamera(
        objPts, imgPtsL, tuple(imgSize), None, None)
    intrinsicL, roi_L = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS +
                       cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Calculate fundamental and essential matrix with stereo calibration
    retStereo, intrinsicL, distL, intrinsicR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objPts, imgPtsL, imgPtsR, intrinsicL, distL, intrinsicR, distR, grayL.shape[::-1], criteria_stereo, flags)

    print("retval:\n", retStereo)
    print("\nRight camera intrinsic matrix:\n", intrinsicR)
    print("\nLeft camera intrinsic matrix:\n", intrinsicL)
    print("\nFundamental:\n", fundamentalMatrix)
    print("\nEssential:\n", essentialMatrix)
    print("\nTranslation:\n", trans)
    print("\nRotation:\n", rot)
    print("\nDistorsion left:\n", distL)
    print("\nDistorsion right:\n", distR)

    return fundamentalMatrix, intrinsicL, intrinsicR, distL, distR
