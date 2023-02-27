from Calibration import CalibrateCamera
from Matching import Matching
from Depth import CalculateDepth

if __name__ == "__main__":
  F, m = CalibrateCamera()
  imgG, imgD, matchedptsG, matchedptsD = Matching(F)  
  CalculateDepth(F,imgG, imgD, matchedptsG, matchedptsD)
