import numpy as np
import cv2 as cv


def CalculateDisparity(fileNumber, intrinsicL, intrinsicR, distL, distR):
    # imgName = "./StereoImages/image" + str(fileNumber) + ".png"
    imgName = "./StereoImages/Center66cm.png"
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
    numDisparities = 192
    blockSize = 1
    disp12MaxDiff = 256
    uniquenessRatio = 2
    speckleWindowSize = 100
    speckleRange = 32

    stereo = cv.StereoSGBM_create(minDisparity=minDisparity,
                                  numDisparities=numDisparities,
                                  blockSize=blockSize,
                                  disp12MaxDiff=disp12MaxDiff,
                                  uniquenessRatio=uniquenessRatio,
                                  speckleWindowSize=speckleWindowSize,
                                  speckleRange=speckleRange,
                                  P1=2*3*blockSize,
                                  P2=32*3*blockSize,
                                  preFilterCap=32,

                                  )
    disparity = stereo.compute(imgLres, imgRres)
    normalized = cv.normalize(disparity, None,
                              0, 255, norm_type=cv.NORM_MINMAX)
    normalized = np.uint8(normalized)
    cv.imwrite("./result/disparity.jpg", cv.hconcat([normalized, imgLres]))

    return normalized
