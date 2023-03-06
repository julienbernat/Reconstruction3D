import numpy as np
import cv2 as cv
from Utils import drawlines
from matplotlib import pyplot as plt


def CalculateDepth(fileNumber, intrinsicL, intrinsicR, distL, distR):
    # imgName = "./StereoImages/image" + str(fileNumber) + ".png"
    imgName = "./StereoImages/Center38cm.png"
    img = cv.imread(imgName, 0)

    imgLres = img[:, :int(len(img[0])/2)]
    imgRres = img[:, int(len(img[0])/2):]

    # Compute undistorted images
    # imgLres = cv.undistort(imgL, intrinsicL, distL, None, None)
    # cv.imwrite("./result/undistortedL.jpg", cv.hconcat([imgL, imgLres]))

    # imgRres = cv.undistort(imgR, intrinsicR, distR, None, None)
    # cv.imwrite("./result/undistortedR.jpg", cv.hconcat([imgR, imgRres]))

    # We compute the depth and disparity map
    minDisparity = 24
    numDisparities = 184
    blockSize = 1
    disp12MaxDiff = 128
    uniquenessRatio = 3
    speckleWindowSize = 1000
    speckleRange = 100
    
    stereo = cv.StereoSGBM_create(minDisparity=minDisparity,
                                  numDisparities=numDisparities,
                                  blockSize=blockSize,
                                  disp12MaxDiff=disp12MaxDiff,
                                  uniquenessRatio=uniquenessRatio,
                                  speckleWindowSize=speckleWindowSize,
                                  speckleRange=speckleRange,
                                  P1=6,
                                  P2=140,
                                  preFilterCap=1,
                                  )
    disparity = stereo.compute(imgLres, imgRres)
    normalized = cv.normalize(disparity, None,
                              0, 255, norm_type=cv.NORM_MINMAX)
    normalized = np.uint8(normalized)
    cv.imwrite("./result/disparity.jpg", cv.hconcat([normalized, imgLres]))

    return normalized
