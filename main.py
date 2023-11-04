import random

import cv2
import keras
import numpy as np
import tensorflow
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Add
from keras.losses import MeanSquaredError
from keras.activations import relu
from os.path import isfile, join
from os import listdir

# Pitati asistenta za pomoć s tom greškom
# Preuzeti datoteke i napraviti učno množico
# Provjeriti parametre - kaj su channels in kaj je format filtra
# https://stackoverflow.com/questions/55324762/the-added-layer-must-be-an-instance-of-class-layer-found-tensorflow-python-ke

# Treba dve veje, ci je stevilo kanalov vhoda neenako stevilu izhoda ReLu

# https://datascience.stackexchange.com/questions/55545/in-cnn-why-do-we-increase-the-number-of-filters-in-deeper-convolution-layers-fo
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

NumpySeznamSlikSpremenjenih = []
NumpySeznamSlikNormalnih = []
# TODO: Sivinska slika s 3 kanale
# TODO: Slika je še vedno samo ena
# TODO: Uporabiti flatten za zadnji sloj?

Direktorij = "E:/Sivinske2014/"
SeznamDatotek = [f for f in listdir(Direktorij) if isfile(join(Direktorij, f))]
print(len(SeznamDatotek))
SeznamPoti = [Direktorij + f for f in SeznamDatotek]


def UstvariOneHotVektor(SeznamZamikov):
	Matrika = [[0] * 21] * 8  # matrika velikost 8 x 21
	OneHotVektor = []
	for j in range(0, 8):
		for i in range(0, 21):
			if RazrediKlasifikacija[i] < SeznamZamikov[j] < RazrediKlasifikacija[i + 1]:
				OneHotVektor = [0] * 21
				OneHotVektor[i] = 1
				break
			Matrika[j] = OneHotVektor
	print(Matrika)


for PotDoSlike in SeznamPoti:
	RGBSivinskaSlika = cv2.imread(PotDoSlike)
	# SivinskaSlika = cv2.cvtColor(SivinskaSlika, cv2.COLOR_BGR2GRAY)
	NumpySeznamSlikNormalnih.append(RGBSivinskaSlika)

	X1Zamik = random.randint(-16, 16)
	X2Zamik = random.randint(-16, 16)
	X3Zamik = random.randint(-16, 16)
	X4Zamik = random.randint(-16, 16)

	Y1Zamik = random.randint(-16, 16)
	Y2Zamik = random.randint(-16, 16)
	Y3Zamik = random.randint(-16, 16)
	Y4Zamik = random.randint(-16, 16)

	K1X = random.randint(16, 128)  # TODO: je nujno 32 zaradi predznaka/ intervala [-16,16], ali je 16 dovolj.
	K1Y = random.randint(16, 128)

	K2X = K1X + 64
	K2Y = K1Y

	K3X = K1X
	K3Y = K1Y + 64

	K4X = K1X + 64
	K4Y = K1Y + 64

	N1X = K1X + X1Zamik
	N1Y = K1Y + Y1Zamik

	N2X = K2X + X2Zamik
	N2Y = K2Y + Y2Zamik

	N3X = K3X + X3Zamik
	N3Y = K3Y + Y3Zamik

	N4X = K4X + X4Zamik
	N4Y = K4Y + Y4Zamik

	block = RGBSivinskaSlika[K1Y:K1Y + 64, K1X:K1X + 64]

	RGBSivinskaSlika[N1X][N1Y] = RGBSivinskaSlika[K1X][K1Y]
	RGBSivinskaSlika[N2X][N2Y] = RGBSivinskaSlika[K2X][K2Y]
	RGBSivinskaSlika[N3X][N3Y] = RGBSivinskaSlika[K3X][K3Y]
	RGBSivinskaSlika[N4X][N4Y] = RGBSivinskaSlika[K4X][K4Y]

	# Display or save the extracted block as needed

	H, status = cv2.findHomography(np.array([(K1X, K1Y), (K2X, K2Y), (K3X, K3Y), (K4X, K4Y)]), np.array([(N1X, N1Y), (N2X, N2Y), (N3X, N3Y), (N4X, N4Y)]))

	InverznaMatrika = np.linalg.inv(H)
	warped_image = cv2.warpPerspective(RGBSivinskaSlika, InverznaMatrika, (320, 240))

	cv2.rectangle(RGBSivinskaSlika, (K1X, K1Y), (K4X, K4Y), (255, 0, 0), 1)
	cv2.rectangle(RGBSivinskaSlika, (N1X, N1Y), (N4X, N4Y), (0, 0, 255), 1)
	KopijaZaRisanje = np.array(RGBSivinskaSlika)
	cv2.imshow('Zamik vizualizacija', RGBSivinskaSlika)
	cv2.imshow('Vhod v NN', warped_image)
	SkupnaSlika = np.concatenate((RGBSivinskaSlika, warped_image), axis=0)
	# SivinskaSlika = cv2.cvtColor(SivinskaSlika, cv2.COLOR_BGR2GRAY)
	SivinskaSlika = cv2.cvtColor(RGBSivinskaSlika, cv2.COLOR_BGR2GRAY)
	SivinskaSlika = tensorflow.expand_dims(SivinskaSlika, axis=0)
	SivinskaSlika = tensorflow.expand_dims(SivinskaSlika, axis=0)
	# print(SkupnaSlika.shape)
	# print('------------------------')
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# im_out = cv2.warpPerspective(SivinskaSlika, h, (im_dst.shape[1], im_dst.shape[0]))

inputShape = (320, 240, 1)

VhodPrvaVeja = keras.layers.Input(shape=inputShape)  # 1. Resnet blok
PolnoPovezanSlojKlasifikacija = Dense(168)(VhodPrvaVeja)
Reshape = tensorflow.reshape(PolnoPovezanSlojKlasifikacija, [8, 21])
Softmax = tensorflow.nn.softmax()

AlternativnaKonvolucija1 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(VhodPrvaVeja)
PrvaKonvolucija1 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(VhodPrvaVeja)
PrviResnetDropOut1 = Dropout(0.5)(PrvaKonvolucija1)
DrugiResnet1 = relu(PrviResnetDropOut1)  # TODO: ReLu sloj?
ZadnjaKonvolucija1 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet1)
DropoutIzhodPrveVeje1 = Dropout(0.5)(ZadnjaKonvolucija1)
SestevekAlternativa1 = Add()([DropoutIzhodPrveVeje1, AlternativnaKonvolucija1])
KoncniSestevek1 = relu(SestevekAlternativa1)

# 2. Resnet blok
AlternativnaKonvolucija2 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek1)
PrvaKonvolucija2 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek1)
PrviResnetDropOut2 = Dropout(0.5)(PrvaKonvolucija2)
DrugiResnet2 = relu(PrviResnetDropOut2)  # TODO: ReLu sloj?
ZadnjaKonvolucija2 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet2)
DropoutIzhodPrveVeje2 = Dropout(0.5)(ZadnjaKonvolucija2)
SestevekAlternativa2 = Add()([DropoutIzhodPrveVeje2, AlternativnaKonvolucija2])
KoncniSestevek2 = relu(PrviResnetDropOut2)

PoolingSloj = MaxPool2D()(KoncniSestevek2)

# 3. Resnet blok
AlternativnaKonvolucija3 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(PoolingSloj)
PrvaKonvolucija3 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(PoolingSloj)
PrviResnetDropOut3 = Dropout(0.5)(PrvaKonvolucija3)
DrugiResnet3 = relu(PrviResnetDropOut3)  # TODO: ReLu sloj?
ZadnjaKonvolucija3 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet3)
DropoutIzhodPrveVeje3 = Dropout(0.5)(ZadnjaKonvolucija3)
SestevekAlternativa3 = Add()([DropoutIzhodPrveVeje3, AlternativnaKonvolucija3])
KoncniSestevek3 = relu(PrviResnetDropOut3)

# 4. Resnet blok
AlternativnaKonvolucija4 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek1)
PrvaKonvolucija4 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek1)
PrviResnetDropOut4 = Dropout(0.5)(PrvaKonvolucija4)
DrugiResnet4 = relu(PrviResnetDropOut4)  # TODO: ReLu sloj?
ZadnjaKonvolucija4 = Conv2D(64, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet4)
DropoutIzhodPrveVeje4 = Dropout(0.5)(ZadnjaKonvolucija4)
SestevekAlternativa4 = Add()([DropoutIzhodPrveVeje4, AlternativnaKonvolucija4])
KoncniSestevek4 = relu(PrviResnetDropOut4)

PoolingSloj = MaxPool2D()(KoncniSestevek4)

# 5. Resnet blok
# VhodPrvaVeja = keras.layers.Input(shape=inputShape)(KoncniSestevek2)
AlternativnaKonvolucija5 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(PoolingSloj)
PrvaKonvolucija5 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(PoolingSloj)
PrviResnetDropOut5 = Dropout(0.5)(PrvaKonvolucija5)
DrugiResnet5 = relu(PrviResnetDropOut5)  # TODO: ReLu sloj?
ZadnjaKonvolucija5 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet5)
DropoutIzhodPrveVeje5 = Dropout(0.5)(ZadnjaKonvolucija5)
SestevekAlternativa5 = Add()([DropoutIzhodPrveVeje5, AlternativnaKonvolucija5])
KoncniSestevek5 = relu(PrviResnetDropOut5)

# 6. Resnet blok
AlternativnaKonvolucija6 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek5)
PrvaKonvolucija6 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek5)
PrviResnetDropOut6 = Dropout(0.5)(PrvaKonvolucija6)
DrugiResnet6 = relu(PrviResnetDropOut6)  # TODO: ReLu sloj?
ZadnjaKonvolucija6 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet6)
DropoutIzhodPrveVeje6 = Dropout(0.5)(ZadnjaKonvolucija6)
SestevekAlternativa6 = Add()([DropoutIzhodPrveVeje6, AlternativnaKonvolucija6])
KoncniSestevek6 = relu(PrviResnetDropOut6)

PoolingSloj = MaxPool2D()(KoncniSestevek6)

AlternativnaKonvolucija7 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(PoolingSloj)
PrvaKonvolucija7 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(PoolingSloj)
PrviResnetDropOut7 = Dropout(0.5)(PrvaKonvolucija7)
DrugiResnet7 = relu(PrviResnetDropOut7)  # TODO: ReLu sloj?
ZadnjaKonvolucija7 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet7)
DropoutIzhodPrveVeje7 = Dropout(0.5)(ZadnjaKonvolucija7)
SestevekAlternativa7 = Add()([DropoutIzhodPrveVeje7, AlternativnaKonvolucija7])
KoncniSestevek7 = relu(PrviResnetDropOut7)

AlternativnaKonvolucija8 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek7)
PrvaKonvolucija8 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(KoncniSestevek7)
PrviResnetDropOut8 = Dropout(0.5)(PrvaKonvolucija8)
DrugiResnet8 = relu(PrviResnetDropOut8)  # TODO: ReLu sloj?
ZadnjaKonvolucija8 = Conv2D(128, (3, 3), activation=tensorflow.keras.activations.relu, padding="same")(DrugiResnet8)
DropoutIzhodPrveVeje8 = Dropout(0.5)(ZadnjaKonvolucija8)
SestevekAlternativa8 = Add()([DropoutIzhodPrveVeje8, AlternativnaKonvolucija8])
KoncniSestevek8 = relu(PrviResnetDropOut8)
# rehspae
PolnoPovezanSloj = Dense(512)(KoncniSestevek8)
print(PolnoPovezanSloj.shape)

# regresijska glava
PolnoPovezanSlojRegresija = Dense(8)(KoncniSestevek8)

# klasifikacijska glava

RazrediKlasifikacija = [-16, -14.4, -12.9, -11.4, -9.60, -8.3, -6.8, -5.3, -3.8, -2.3, -0.8,  0.8, 2.3, 3.8, 5.3, 6.8, 8.3, 9.9, 11.43, 12.9, 14.4, 16.0]
model = keras.Model(
	inputs=[VhodPrvaVeja],
	outputs=[PolnoPovezanSloj],
)

model.build(input_shape=(320, 240, 1))
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics="acc")  # https://machinelearningmastery.com/loss-functions-in-tensorflow/
print(model.summary())
