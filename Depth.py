import numpy as np
import cv2 as cv
from Utils import drawlines
from matplotlib import pyplot as plt


def CalculateDisparity(fileNumber, intrinsicL, intrinsicR, distL, distR):
    imgName = "./StereoImages/image" + str(fileNumber) + ".png"
    img = cv.imread(imgName, 0)

    moitier = len(img[0])/2
    imgL = img[:, :int(moitier)]
    imgR = img[:, int(moitier):]
    # imgL = cv.imread("./StereoImages/tsukuba_l.png")
    # imgR = cv.imread("./StereoImages/tsukuba_r.png")
    # Compute undistorted images
    imgLres = cv.undistort(imgL, intrinsicL, distL, None, None)
    cv.imwrite("./result/undistortedL.jpg", cv.hconcat([imgL, imgLres]))

    imgRres = cv.undistort(imgR, intrinsicR, distR, None, None)
    cv.imwrite("./result/undistortedR.jpg", cv.hconcat([imgR, imgRres]))

    # We compute the depth and disparity map
    minDisparity = 16
    # maxDisparity = 128
    maxDisparity = minDisparity * 9
    numDisparities = maxDisparity-minDisparity
    blockSize = 3
    disp12MaxDiff = 36
    uniquenessRatio = 4
    speckleWindowSize = 10000
    speckleRange = 100
    
    stereo = cv.StereoSGBM_create(minDisparity=minDisparity,
                                  numDisparities=numDisparities,
                                  blockSize=blockSize,
                                  disp12MaxDiff=disp12MaxDiff,
                                  uniquenessRatio=uniquenessRatio,
                                  speckleWindowSize=speckleWindowSize,
                                  speckleRange=speckleRange,
                                  P1=16 * 3 * blockSize * blockSize,
                                  P2=64 * 3 * blockSize * blockSize,
                                  preFilterCap=30,
                                  )
    disparity = stereo.compute(imgL, imgR)
    # disparity = cv.normalize(disparity, disparity,
    #                          alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    # disparity = np.uint8(disparity)
    f = intrinsicL[1,1]

    depth = 0.06 * f / disparity
    plt.imshow(depth, 'gray')
    plt.show()
    # h, w = imgL.shape[:2]
    # f = intrinsicL[1,1]
    # Q = np.float32([[1, 0, 0, -0.5*w],
    #                 [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
    #                 [0, 0, 0,     -f], # so that y-axis looks up
    #                 [0, 0, 1,      0]])
    # points_3D = cv.reprojectImageTo3D(disparity, Q)
    # print(points_3D)
    return disparity
