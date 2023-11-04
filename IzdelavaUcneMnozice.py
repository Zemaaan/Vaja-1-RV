from os import listdir
from os.path import isfile, join

import cv2
from PIL import Image

Direktorij = "E:/train2014/"
SeznamDatotek = [f for f in listdir(Direktorij) if isfile(join(Direktorij, f))]
print(SeznamDatotek)
SeznamPoti = [Direktorij + f for f in SeznamDatotek]
indeks = 0
for PotDoSlike in SeznamPoti:
	img_rgb = cv2.imread(PotDoSlike)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.resize(img_gray, (320, 240))
	# print(img_gray)
	cv2.imwrite('E:/Sivinske2014/' + SeznamDatotek[indeks], img_gray)
	indeks += 1

