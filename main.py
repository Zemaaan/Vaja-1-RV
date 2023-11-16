import random
import cv2
import keras
import numpy
import numpy as np
import tensorflow
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Add
from keras.losses import MeanSquaredError
from keras.activations import relu
from tensorflow.keras.utils import split_dataset
from os.path import isfile, join
from os import listdir
import copy
RazrediKlasifikacija = [-16, -14.4, -12.9, -11.4, -9.60, -8.3, -6.8, -5.3, -3.8, -2.3, -0.8,  0.8, 2.3, 3.8, 5.3, 6.8, 8.3, 9.9, 11.43, 12.9, 14.4, 16.0]

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
SeznamPoti = [Direktorij + f for f in SeznamDatotek]

def UstvariOneHotVektor(SeznamZamikov):
	Matrika = [[0] * 21] * 8  # matrika velikost 8 x 21
	OneHotVektor = []
	for j in range(0, 8):
		for i in range(0, 21):
			if RazrediKlasifikacija[i] < SeznamZamikov[j] < RazrediKlasifikacija[i + 1]:
				OneHotVektor = [0] * 21
				OneHotVektor[i] = 1
			Matrika[j] = OneHotVektor
		# print(Matrika[j])
	return Matrika

features = []
labels = []

features = np.array(features)
labels = np.array(labels)

for PotDoSlike in SeznamPoti[:10]:
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

	K1X = random.randint(16, 150)  # TODO: je nujno 32 zaradi predznaka/ intervala [-16,16], ali je 16 dovolj.
	K1Y = random.randint(16, 150)

	K2X = K1X + 64
	K2Y = K1Y

	K3X = K2X
	K3Y = K2Y + 64

	K4X = K3X - 64
	K4Y = K3Y

	N1X = K1X + X1Zamik
	N1Y = K1Y + Y1Zamik

	N2X = K2X + X2Zamik
	N2Y = K2Y + Y2Zamik

	N3X = K3X + X3Zamik
	N3Y = K3Y + Y3Zamik

	N4X = K4X + X4Zamik
	N4Y = K4Y + Y4Zamik

	block = RGBSivinskaSlika[K1X:K1X + 64, K1Y:K1Y + 64]

	RGBSivinskaSlika[N1X][N1Y] = RGBSivinskaSlika[K1X][K1Y]
	RGBSivinskaSlika[N2X][N2Y] = RGBSivinskaSlika[K2X][K2Y]
	RGBSivinskaSlika[N3X][N3Y] = RGBSivinskaSlika[K3X][K3Y]
	RGBSivinskaSlika[N4X][N4Y] = RGBSivinskaSlika[K4X][K4Y]

	# Display or save the extracted block as needed

	H, status = cv2.findHomography(np.array([(K1X, K1Y), (K2X, K2Y), (K3X, K3Y), (K4X, K4Y)]), np.array([(N1X, N1Y), (N2X, N2Y), (N3X, N3Y), (N4X, N4Y)]))  # TODO: Manjka slika

	InverznaMatrika = np.linalg.inv(H)
	warped_image = cv2.warpPerspective(RGBSivinskaSlika, InverznaMatrika, (320, 240))  # TODO: H^-1 ali enostavno H za prvo homografijo?

	SlikaZaBarvanjePopacana = copy.deepcopy(warped_image)
	SlikaZaBarvanjeOriginalna = copy.deepcopy(RGBSivinskaSlika)

	cv2.line(SlikaZaBarvanjeOriginalna, (K1X, K1Y), (K2X, K2Y), (255, 0, 0), 1)  # rdeca je zamik, modra je original
	cv2.line(SlikaZaBarvanjeOriginalna, (K2X, K2Y), (K3X, K3Y), (255, 0, 0), 1)
	cv2.line(SlikaZaBarvanjeOriginalna, (K3X, K3Y), (K4X, K4Y), (255, 0, 0), 1)
	cv2.line(SlikaZaBarvanjeOriginalna, (K4X, K4Y), (K1X, K1Y), (255, 0, 0), 1)

	cv2.line(SlikaZaBarvanjeOriginalna, (N1X, N1Y), (N2X, N2Y), (0, 0, 255), 1)
	cv2.line(SlikaZaBarvanjeOriginalna, (N2X, N2Y), (N3X, N3Y), (0, 0, 255), 1)
	cv2.line(SlikaZaBarvanjeOriginalna, (N3X, N3Y), (N4X, N4Y), (0, 0, 255), 1)
	cv2.line(SlikaZaBarvanjeOriginalna, (N4X, N4Y), (N1X, N1Y), (0, 0, 255), 1)
	# matplotlib

	cv2.line(SlikaZaBarvanjePopacana, (N1X, N1Y), (N2X, N2Y), (0, 0, 255), 1)
	cv2.line(SlikaZaBarvanjePopacana, (N2X, N2Y), (N3X, N3Y), (0, 0, 255), 1)
	cv2.line(SlikaZaBarvanjePopacana, (N3X, N3Y), (N4X, N4Y), (0, 0, 255), 1)
	cv2.line(SlikaZaBarvanjePopacana, (N4X, N4Y), (N1X, N1Y), (0, 0, 255), 1)

	cv2.line(SlikaZaBarvanjePopacana, (K1X, K1Y), (K2X, K2Y), (255, 0, 0), 1)
	cv2.line(SlikaZaBarvanjePopacana, (K2X, K2Y), (K3X, K3Y), (255, 0, 0), 1)
	cv2.line(SlikaZaBarvanjePopacana, (K3X, K3Y), (K4X, K4Y), (255, 0, 0), 1)
	cv2.line(SlikaZaBarvanjePopacana, (K4X, K4Y), (K1X, K1Y), (255, 0, 0), 1)

	VhodNormalnaSlika = cv2.cvtColor(RGBSivinskaSlika, cv2.COLOR_BGR2GRAY)
	VhodPopacanaSlika = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

	VhodNormalnaSlika = VhodNormalnaSlika[K1X:K1X + 64, K1Y:K1Y + 64]
	VhodPopacanaSlika = VhodPopacanaSlika[K1X:K1X + 64, K1Y:K1Y + 64]

	# VhodNormalnaSlika = cv2.resize(VhodNormalnaSlika, (320, 240), cv2.INTER_LINEAR)
	# VhodPopacanaSlika = cv2.resize(VhodPopacanaSlika, (320, 240), cv2.INTER_LINEAR)

	cv2.imshow('Vhod v NN - Normalna', SlikaZaBarvanjeOriginalna)
	cv2.imshow('Vhod v NN - Popacana', SlikaZaBarvanjePopacana)

	SkupnaSlika = np.concatenate((VhodNormalnaSlika, VhodPopacanaSlika), 1)  # SivinskaSlika = cv2.cvtColor(SivinskaSlika, cv2.COLOR_BGR2GRAY)
	SkupnaSlika = tensorflow.reshape(SkupnaSlika, [64, 64, 2])

	UcnaMatrika = UstvariOneHotVektor(SeznamZamikov=[X1Zamik, Y1Zamik, X2Zamik, Y2Zamik, X3Zamik, Y3Zamik, X4Zamik, Y4Zamik])

	np.append(features, SkupnaSlika)
	np.append(labels, UcnaMatrika)


	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# im_out = cv2.warpPerspective(SivinskaSlika, h, (im_dst.shape[1], im_dst.shape[0]))



print(features.shape)
inputShape = (64, 64, 2)

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


model = keras.Model(
	inputs=[VhodPrvaVeja],
	outputs=[PolnoPovezanSloj],
)

model.build(input_shape=(320, 240, 1, 1))
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics="acc")  # https://machinelearningmastery.com/loss-functions-in-tensorflow/
(x_train, y_train), (x_test, y_test) = train_test_split(features, labels)
model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

# Kak dobimo "poravnano" sliko?
# Katero matriko uporabljamo?
