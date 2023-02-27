import numpy as np
import cv2 as cv

def drawdot(img1,img2,pts1,pts2):
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def drawdotline(img1, img2, pts1, pts2):
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for indx in range(len(pts1)):
    
        color = tuple(np.random.randint(0,255,3).tolist())
        p1 = pts1[indx]
        p2 = pts2[indx]
        img1 = cv.circle(img1,p1,5,color,-1)
        img2 = cv.circle(img2,p2,5,color,-1)
        img1 = cv.line(img1, p1, p2, color,1)
    return img1, img2

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
  
def ConstructWindow(img, coordo, sizeT):
    #reduit la taille de la fenetre sur les bords
    Xlow = max(coordo[0]-sizeT, 0)
    Xhigh = min(coordo[0]+ sizeT, len(img[0]))
    Ylow = max(coordo[1]-sizeT, 0)
    Yhigh = min(coordo[1]+ sizeT, len(img[0]))


    template = img[Ylow:Yhigh, Xlow:Xhigh]
    return template
