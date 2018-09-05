import numpy as np
import cv2
from matplotlib import pyplot as plt

import glob

for img_name in glob.glob('capture*.jpg'):
  dt = img_name.replace('capture', '').replace('.jpg', '')
  print("Handling", dt, img_name)
  img = cv2.imread(img_name)

  for denoiselv in range(5, 25, 5):
    for lightscale in range(1, 10, 1):
      print("doing...", denoiselv, lightscale)
      dst = cv2.fastNlMeansDenoisingColored(img, None, denoiselv, denoiselv, 7, 21)

      hsvold = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      hsvnew = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
      hsvold[..., 2] = hsvold[..., 2]*(lightscale/10.0)
      hsvnew[..., 2] = hsvnew[..., 2]*(lightscale/10.0)

      cv2.imwrite( dt + "_denoise_" + str(denoiselv) + ".jpg", dst)
      cv2.imwrite( dt + "_light_" + str(lightscale) + ".jpg", cv2.cvtColor(hsvold, cv2.COLOR_HSV2BGR))
      cv2.imwrite( dt + "_denoise_" + str(denoiselv) + "_light_" + str(lightscale) + ".jpg", cv2.cvtColor(hsvnew, cv2.COLOR_HSV2BGR))
      print("done denoising")
