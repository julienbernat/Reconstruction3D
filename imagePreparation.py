import cv2 as cv
import numpy as np

def ImagePreparation(img):
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
  contours, hierarchy, a = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cnt = contours[0]
  x, y, w, h = cv.boundingRect(cnt)
  cropped = img[y + 5 : y + h - 5, x + 5 : x + w - 5]
  cv.imwrite('result/cropped.png',cropped)
  return cropped
