import numpy as np
import cv2 as cv


def CalculateDisparity(img, intrinsicL,cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR):
    # imgName = "./StereoImages/image" + str(fileNumber) + ".png"


    imgL = img[:, :int(len(img[0])/2)]
    imgR = img[:, int(len(img[0])/2):]

    # Compute undistorted images
    imgLres = cv.undistort(imgL, cameraMatrixL, distL, None, intrinsicL)
    cv.imwrite("./result/undistortedL.jpg", imgLres)

    imgRres = cv.undistort(imgR, cameraMatrixR, distR, None, intrinsicR)
    cv.imwrite("./result/undistortedR.jpg",imgRres)

    # laplacianL = cv.Laplacian(imgLres,cv.CV_64F)
    # laplacianR = cv.Laplacian(imgRres,cv.CV_64F)
    # laplacian = cv.Laplacian(img,cv.CV_64F)

    # sobelR = cv.Sobel(imgLres,cv.CV_64F,1,1,ksize=1)
    # sobelL = cv.Sobel(imgRres,cv.CV_64F,1,1,ksize=1)
    # sobel = cv.Sobel(img,cv.CV_64F,1,1,ksize=1)
    
    # cv.imwrite("./result/laplacian.jpg", laplacian)
    # cv.imwrite("./result/laplacianL.jpg", laplacianL)
    # cv.imwrite("./result/laplacianR.jpg", laplacianR)

    # cv.imwrite("./result/sobel.jpg", sobel)
    # cv.imwrite("./result/sobelL.jpg", sobelL)
    # cv.imwrite("./result/sobelR.jpg", sobelR)

    # normalizedL = cv.normalize(sobelL, None,
    #                           0, 255, norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
    # normalizedR = cv.normalize(sobelR, None,
    #                           0, 255, norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)

    # grayL = cv.cvtColor(normalizedL,cv.COLOR_BGR2GRAY)
    # grayR = cv.cvtColor(normalizedR,cv.COLOR_BGR2GRAY)



    # We compute the depth and disparity map
    minDisparity = 24
    numDisparities = 192
    blockSize = 1
    disp12MaxDiff = 256
    uniquenessRatio = 2
    speckleWindowSize = 200
    speckleRange = 32

    stereo = cv.StereoSGBM_create(minDisparity=minDisparity,
                                  numDisparities=numDisparities,
                                  blockSize=blockSize,
                                  P1=2*3*blockSize,
                                  P2=32*3*blockSize,
                                  disp12MaxDiff=disp12MaxDiff,
                                  uniquenessRatio=uniquenessRatio,
                                  speckleWindowSize=speckleWindowSize,
                                  speckleRange=speckleRange,
                                  preFilterCap=32,
                                  mode=cv.STEREO_SGBM_MODE_SGBM)

    disparity = stereo.compute(imgLres, imgRres)
    # disparity = stereo.compute(grayL, grayR)
    # cv.imwrite("./result/disparity.jpg", disparity)
    normalized = cv.normalize(disparity, None,
                              0, 255, norm_type=cv.NORM_MINMAX)
    # cv.imwrite("./result/normalized.jpg", normalized)
    normalized = np.uint8(normalized)
    cv.imwrite("./result/depth.jpg", normalized)
    blur = cv.medianBlur(normalized,7)
    cv.imwrite("./result/blur.jpg", blur)
    return blur
