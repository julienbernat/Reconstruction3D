from Calibration import CalibrateCamera
from Matching import Matching
from Depth import CalculateDepth
import argparse

parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("-f", "--file_number", help="Prints the supplied argument.", type=int)

if __name__ == "__main__":
  fundamental = CalibrateCamera()
  imgG, imgD, matchedptsG, matchedptsD = Matching(fundamental, parser.parse_args().file_number)  
  CalculateDepth(fundamental,imgG, imgD, matchedptsG, matchedptsD)
