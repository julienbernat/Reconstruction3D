from Calibration import CalibrateCamera
from Matching import Matching
from Depth import CalculateDisparity

import argparse

parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("-f", "--file_number",
                    help="Prints the supplied argument.", type=int)


if __name__ == "__main__":
    fundamental, intrinsicL, intrinsicR, distL, distR = CalibrateCamera()
    imgG, imgD, matches, = Matching(
        fundamental, parser.parse_args().file_number)
    
    CalculateDisparity(parser.parse_args().file_number,
                   intrinsicL, intrinsicR, distL, distR)
    

    # imgL = cv.imread('./StereoImages/tsukuba_l.png', cv.IMREAD_GRAYSCALE)
    # imgR = cv.imread('./StereoImages/tsukuba_r.png', cv.IMREAD_GRAYSCALE)
    # stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    # disparity = stereo.compute(imgL,imgR)

    # h, w = imgL.shape[:2]
    # f = 0.8 * w  # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5 * w],
    #                 [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0, -f],  # so that y-axis looks up
    #                 [0, 0, 1, 0]])
    # points = cv.reprojectImageTo3D(disparity, Q)
    # colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    # mask = disparity > disparity.min()
    # out_points = points[mask]
    # out_colors = colors[mask]
    # disparity_scaled = (disparity - 16) / 32
    # disparity_scaled += abs(np.amin(disparity_scaled))
    # disparity_scaled /= np.amax(disparity_scaled)
    # disparity_scaled[disparity_scaled < 0] = 0
    # depth = np.array(255 * disparity_scaled, np.uint8)
    # plt.imshow(depth,'gray')
    # plt.show()


