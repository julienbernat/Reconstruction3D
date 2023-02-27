import numpy as np
import cv2 as cv
from Utils import drawlines

def CalculateDepth(F,imgG, imgD, matchedptsG, matchedptsD):
    pointsG = np.array(matchedptsG)
    pointsD = np.array(matchedptsD)
    n = len(pointsG)
    pt1 = np.reshape(pointsG,(1,n,2))
    pt2 = np.reshape(pointsD,(1,n,2))   

    p1, p2 = cv.correctMatches(F,pt1, pt2)

    lines = cv.computeCorrespondEpilines(pointsG.reshape(-1,1,2), 2,F)
    lines = lines.reshape(-1,3)
    resG,resD = drawlines(imgG,imgD,lines,pointsG,matchedptsD)
    cv.imwrite("./result/droiteepipolaire.jpg", cv.hconcat([resG, resD]))
    return
