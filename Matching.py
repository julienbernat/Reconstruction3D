import numpy as np
import cv2 as cv
from Utils import ConstructWindow, drawdot, drawdotline


def Matching(F, fileNumber):
    test = "./StereoImages/image" + str(fileNumber) + ".png"

    confidenceBound = 0.35
    windowsize = 60 
    dispariteMax = 500

    img = cv.imread(test, 0)

    moitier = len(img[0])/2
    imgG = img[:, :int(moitier)]
    imgD = img[:, int(moitier):]

    edgesG = cv.Canny(imgG, 120, 120)
    edgesD = cv.Canny(imgD, 120, 120)

    cv.imwrite("./result/filtreCanny.jpg", cv.hconcat([edgesG, edgesD]))

    sift = cv.SIFT_create()
    keypointsG, waste1 = sift.detectAndCompute(edgesG, None)
    keypointsD, waste2 = sift.detectAndCompute(edgesD, None)

    coordoPtsG = np.int32([k.pt for k in keypointsG])
    coordoPtsD = np.int32([k.pt for k in keypointsD])
    nbPtsG = len(coordoPtsG)
    nbPtsD = len(coordoPtsD)
    print("found (", nbPtsG,",",nbPtsD, ") points d'interet")
    print("for example: ", coordoPtsD[nbPtsD-3])

    imgggg, imgddd = drawdot(edgesG, edgesD, coordoPtsG, coordoPtsD)
    imggg, imgdd = drawdot(imgG, imgD, coordoPtsG, coordoPtsD)
    cv.imwrite("./result/pointsInteretCanny.jpg", cv.hconcat([imgggg, imgddd]))
    cv.imwrite("./result/pointsdinteresimg.jpg", cv.hconcat([imggg, imgdd]))

    matchedptsG, matchedptsD, aberrantG, aberrantD = FindMatches(imgG, imgD, coordoPtsG, coordoPtsD, windowsize,dispariteMax,confidenceBound)

    resG, resD = drawdotline(imgG, imgD, matchedptsG,matchedptsD)
    disG, disD = drawdotline(imgG, imgD, aberrantG, aberrantD)
    cv.imwrite("./result/correlationmatchin.jpg", cv.hconcat([resG, resD]))
    cv.imwrite("./result/correlationaberant.jpg", cv.hconcat([disG, disD]))

    cv.destroyAllWindows()

    return imgG, imgD, matchedptsG, matchedptsD

def FindSinglePoint(imgG, imgD, pt, pointsD, boxsize, disparityMax):
    # pour chaque poits Gauche
    # creer une sousmatrice qui inclus le voisinage

    template = ConstructWindow(imgG, pt, boxsize)
        
    
    # evaluer tous les points Droits
    maxCor = 0
    match = (0,0)

    for candidat in pointsD:

        if (abs(pt[1] - candidat[1]) < 30) and (abs(pt[0] - candidat[0]) < disparityMax):
                
            sousFenetre = ConstructWindow(imgD, candidat, boxsize)
            if(len(template) > len(sousFenetre)) or len(template[0]) > len(sousFenetre[0]):
                tempTemplate = ConstructWindow(imgG, pt, min(int(len(sousFenetre)/2),int(len(sousFenetre[0])/2)))


                corMatrice = cv.matchTemplate(sousFenetre, tempTemplate, cv.TM_CCORR_NORMED)
            else:    
                corMatrice = cv.matchTemplate(sousFenetre, template, cv.TM_CCORR_NORMED)
            cor = max(map(max, corMatrice))
            if cor > maxCor:
                maxCor = cor
                match = candidat
    return match, maxCor

def FindMatches(imgG, imgD, pointsG, pointsD, boxsize, disparityMax, confidenceBound):
    matchedptsD = []
    matchedptsG = []
    abberantD = []
    abberantG = []
    it = 0
    nb = 0
    for pt in pointsG:
        it += 1
        #if(it%15==0):
            #print(it)

        match, maxCor = FindSinglePoint(imgG, imgD, pt, pointsD, boxsize, disparityMax)

        #garder le maximum si un maximum est resonable
        if(abs(pt[1] - match[1]) < 30) and (abs(pt[0] - match[0]) < disparityMax) and maxCor > confidenceBound:
            matchedptsG.append(pt)
            matchedptsD.append(match)
            nb+=1
            if(nb%100==0):
                print("found ", nb, "matches so far")
        else:
            abberantG.append(pt)
            abberantD.append(match)
        
    cv.destroyAllWindows()
    np.int32(matchedptsD)
    np.int32(matchedptsG)
    np.int32(abberantG)
    np.int32(abberantD)

    return matchedptsG, matchedptsD, abberantG, abberantD


