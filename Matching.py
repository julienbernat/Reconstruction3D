import numpy as np
import cv2 as cv
from Utils import drawlines
from Validation import MatchingValidation


def Matching(F, fileNumber):
    # Read image
    # imgName = "./StereoImages/image" + str(fileNumber) + ".png"
    imgName = "./StereoImages/Center38cm.png"
    img = cv.imread(imgName)

    imgL = img[:, :int(len(img[0])/2)]
    imgR = img[:, int(len(img[0])/2):]

    # We use sift to calculate key points
    sift = cv.SIFT_create()

    # We can detect key points from the original image
    kpL, desL = sift.detectAndCompute(imgL, None)
    kpR, desR = sift.detectAndCompute(imgR, None)

    # We draw keypoints over the original image
    imgkpL = cv.drawKeypoints(imgL, kpL, 0, (0, 0, 255), None)
    imgkpR = cv.drawKeypoints(imgR, kpR, 0, (0, 0, 255), None)
    cv.imwrite("./result/keypoints.jpg", cv.hconcat([imgkpL, imgkpR]))

    # # We match key points together
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desL, desR, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])

    drawMatches = cv.drawMatchesKnn(imgL, kpL, imgR, kpR, good, None, flags=2)
    cv.imwrite("./result/keypoints.jpg", drawMatches)

    # matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    # knn_matches = matcher.knnMatch(desL, desR, 2)
    # ratio_thresh = 0.5
    # good = []
    # for m,n in knn_matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         good.append(m)

    # img_matches = np.empty((max(imgL.shape[0], imgR.shape[0]), imgL.shape[1]+imgR.shape[1], 3), dtype=np.uint8)
    # drawMatches = cv.drawMatches(imgL, kpL, imgR, kpR, good, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv.imwrite("./result/keypoints.jpg", drawMatches)

    ptsR = []
    ptsL = []

    for match in good:
        pL = kpL[match[0].queryIdx].pt
        pR = kpR[match[0].trainIdx].pt

        ptsR.append(pR)
        ptsL.append(pL)

    # We compute Epilines
    lines1 = cv.computeCorrespondEpilines(
        np.array(ptsL).reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    lines2 = cv.computeCorrespondEpilines(
        np.array(ptsR).reshape(-1, 1, 2), 2, F)
    lines2 = lines2.reshape(-1, 3)

   # MatchingValidation(ptsR, ptsL, lines1, lines2, F)

    epilines, original = drawlines(cv.cvtColor(imgL, cv.COLOR_BGR2GRAY), cv.cvtColor(
        imgR, cv.COLOR_BGR2GRAY), lines1, np.int32(ptsL), np.int32(ptsR))

    cv.imwrite("./result/epipolarlines.jpg",
               cv.hconcat([epilines, original]))

    return imgL, imgR, good
