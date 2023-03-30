import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

def CalculateEyeDepth(pixels, depth):
    res=[]

    x= [p[0] for p in pixels]
    y= [p[1] for p in pixels]


    horx = np.linspace(min(x), max(x), 15)
    hory = np.full(15, min(y)+0.5*(max(y)-min(y)))

    verty = np.linspace(min(y), max(y), 8)
    vertx = np.full(8, min(x)+0.5*(max(x)-min(x)))

    x= np.append(x,horx)
    y = np.append(y,hory)

    x = np.append(x,vertx)
    y = np.append(y,verty)

    new_points = [(x[i], y[i]) for i in range(len(x))]
    for point in new_points:
        res.append(round(depth[int(point[0])][int(point[1])],3))
        cv.circle(depth, (int(point[0]), int(point[1])), radius=1, color=(255, 255, 255), thickness=3)

    median=np.median(res)
    average=np.average(res)

    return median

def DownsizeImage(imgL,imgR):
    if len(imgL.shape) > 2:
        col, row = imgL.shape[:2]
    else:
        col, row = imgL.shape

    imgL_downSampled = cv.resize(imgL, dsize=(row // 2, col // 2))
    imgR_downSampled = cv.resize(imgR, dsize=(row //2, col // 2))

    return imgL_downSampled, imgR_downSampled

def Laplacian(imgL,imgR):
    laplacianL = cv.Laplacian(imgL,cv.CV_64F)
    laplacianR = cv.Laplacian(imgR,cv.CV_64F)

    cv.imwrite("./result/laplacianL.jpg", laplacianL)
    cv.imwrite("./result/laplacianR.jpg", laplacianR)

def Sobel(imgL,imgR):
    sobelR = cv.Sobel(imgL,cv.CV_64F,1,1,ksize=1)
    sobelL = cv.Sobel(imgR,cv.CV_64F,1,1,ksize=1)

    cv.imwrite("./result/sobelL.jpg", sobelL)
    cv.imwrite("./result/sobelR.jpg", sobelR)

def Undistort(imgL,imgR,cameraMatrixL, distL, intrinsicL, cameraMatrixR, distR, intrinsicR):
    # We compute the undistorted images
    imgLres = cv.undistort(imgL, cameraMatrixL, distL, None, intrinsicL)
    cv.imwrite("./result/undistortedL.jpg", imgLres)

    imgRres = cv.undistort(imgR, cameraMatrixR, distR, None, intrinsicR)
    cv.imwrite("./result/undistortedR.jpg",imgRres)

    return imgLres, imgRres

def Disparity(imgL,imgR):
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

    disparity = stereo.compute(imgL, imgR)
    normalized = cv.normalize(disparity, None,1, 255, norm_type=cv.NORM_MINMAX)
    normalized = np.uint8(normalized)
    cv.imwrite("./result/disparity.jpg", normalized)
    return disparity

def Depth(disparity, cameraMatrixL, leftEyePixels, rightEyePixels):
    # We compute the depth map
    f = cameraMatrixL[0, 0]
    depth = (f*60) / cv.blur(disparity, (10, 10))
    # depth = np.clip(depth, 10, 60)
    normalized = cv.normalize(depth, None,1, 255, norm_type=cv.NORM_MINMAX)
    normalized = np.uint8(normalized)

    distLeftEye = CalculateEyeDepth(leftEyePixels,depth)
    distRightEye = CalculateEyeDepth(rightEyePixels,depth)

    font = cv.FONT_HERSHEY_SIMPLEX

    cv.putText(depth, 'left eye : ' + str(round(distLeftEye, 3)) + " cm", (10,450), font, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(depth, 'right eye : ' + str(round(distRightEye, 3)) + " cm", (10,500), font, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(depth, 'average : ' + str(round(0.5*(distLeftEye+distRightEye), 3)) + " cm", (10,550), font, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imwrite("./result/depth.jpg", normalized)
    cv.imwrite("./result/depthWithEyes.jpg", depth)
    return depth


def CalculateDepth(img, intrinsicL, cameraMatrixL, intrinsicR, cameraMatrixR, distL, distR, leftEyePixels, rightEyePixels):
    imgL = img[:, :int(len(img[0])/2)]
    imgR = img[:, int(len(img[0])/2):]

    imgL, imgR = Undistort(imgL,imgR,cameraMatrixL, distL, intrinsicL, cameraMatrixR, distR, intrinsicR)

    disparity = Disparity(imgL, imgR)

    depth = Depth(disparity, cameraMatrixL, leftEyePixels, rightEyePixels)

    # fig, ax = plt.subplots()
    # im = ax.imshow(depth, interpolation='none')
    # ax.format_coord = Formatter(im)
    # plt.show()

    return depth
