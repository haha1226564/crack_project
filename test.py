import keras
from keras.applications.xception import Xception
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os

basemodel = Xception(include_top=False, weights='imagenet')
inputs = Input(shape=(100, 100, 3))

x = basemodel(inputs)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.1)(x)
y = Dense(1, activation='sigmoid')(x)

basemodel.trainable = False
model = Model(inputs=inputs, outputs=y)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['acc'])


model.load_weights("weight.hdf5")




#os.remove("temp/t/a.jpg")

for i in os.listdir("ground_crack_samples"):
	img = cv2.imread("ground_crack_samples/" + i)

	h = 50

	for x in range(0, 480-h, h):
		for y in range(0, 320-h, h):
			cv2.rectangle(img, (x, y), (x+h, y+h), (0,128,0), 1)
			crop_img = img[x:x+h, y:y+h]
			cv2.imwrite("temp/t/a.jpg", crop_img)

			predictGenerator = ImageDataGenerator(rescale=1./255)
			traindata = predictGenerator.flow_from_directory("temp/", target_size=(100, 100), batch_size=64, class_mode='binary', shuffle=False)
			y = model.predict_generator(traindata)

			if y[0] > 0.5:
				cv2.rectangle(img, (x, y), (x+h, y+h), (255,0,0), 3)

			cv2.imshow("1", img)
			cv2.waitKey()

			os.remove("temp/t/a.jpg")

	cv2.imshow("1", img)
	cv2.waitKey()
	cv2.imwrite("ground_crack_sample_out/" + i, img)







