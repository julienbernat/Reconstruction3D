import numpy as np
import cv2 as cv
import glob
from Validation import CalibrationValidation
from imagePreparation import ImagePreparation


def CalibrateCamera():
    stereoChessboards = glob.glob("./StereoChessboards/*")

    # Chessboard size
    cbSize = (5, 6)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D points of chessboard
    objPtsL = []
    objPtsR = []

    # 2D points of left and right image
    imgPtsL = []
    imgPtsR = []

    # World coordinates for 3D points
    coord = np.zeros((cbSize[0]*cbSize[1], 3), np.float32)
    coord[:, :2] = np.mgrid[0:cbSize[0], 0:cbSize[1]].T.reshape(-1, 2)

    for i,chessBoard in enumerate(stereoChessboards):
        img = cv.imread(chessBoard)
        cropped = ImagePreparation(img)

        # Separate left and right image
        imgL = cropped[:, :int(len(cropped[0])/2)]
        imgR = cropped[:, int(len(cropped[0])/2):]

        # Find corners of the chessboard
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        retL, cornersL = cv.findChessboardCorners(grayL, cbSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        retR, cornersR = cv.findChessboardCorners(grayR, cbSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)


        if retL == True:
            objPtsL.append(coord)

            # Add 2d image points of the chessboard corners
            cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgPtsL.append(cornersL)
            cv.drawChessboardCorners(imgL, cbSize, cornersL, retL)

        if retR == True:
            objPtsR.append(coord)

            # Add 2d image points of the chessboard corners
            cv.cornerSubPix(grayR, cornersR, (2, 2), (-1, -1), criteria)
            imgPtsR.append(cornersR)
            cv.drawChessboardCorners(imgR, cbSize, cornersR, retR)

        cv.imwrite("./result/ChessboardCorners"+str(i)+".jpg", cv.hconcat([imgL, imgR]))

    # Calculate intrisic  matrices
    heightL, widthL, channelsL = imgL.shape
    print(widthL)
    print(heightL)
    retval, cameraMatrixL, distL, rvecs, tvecs = cv.calibrateCamera(
        objPtsL, imgPtsL, (widthL,heightL), None, None)
    intrinsicL, roi_L = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, (widthL,heightL), 1, (widthL,heightL))

    heightR, widthR, channelsR = imgR.shape
    print(widthR)
    print(heightR)
    retval, cameraMatrixR, distR, rvecs, tvecs = cv.calibrateCamera(
        objPtsR, imgPtsR, (widthR,heightR), None, None)
    intrinsicR, roi_R = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, (widthR,heightR), 1, (widthR,heightR))


    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS +
                       cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Calculate fundamental and essential matrix with stereo calibration
    retStereo, intrinsicL, distL, intrinsicR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objPtsL, imgPtsL, imgPtsR, intrinsicL, distL, intrinsicR, distR, grayL.shape[::-1], criteria_stereo, flags)

    CalibrationValidation(intrinsicL, intrinsicR, rot,
                          trans, essentialMatrix, fundamentalMatrix)
    # print("retval:\n", retStereo)
    # print("\nRight camera intrinsic matrix:\n", intrinsicR)
    # print("\nLeft camera intrinsic matrix:\n", intrinsicL)
    # print("\nFundamental:\n", fundamentalMatrix)
    # print("\nEssential:\n", essentialMatrix)
    # print("\nTranslation:\n", trans)
    # print("\nRotation:\n", rot)
    # print("\nDistorsion left:\n", distL)
    # print("\nDistorsion right:\n", distR)

    return fundamentalMatrix, intrinsicL,cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR
