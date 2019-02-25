import keras
from keras.applications.xception import Xception
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

basemodel = Xception(include_top=False, weights='imagenet')
inputs = Input(shape=(100, 100, 3))

x = basemodel(inputs)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
y = Dense(1, activation='sigmoid')(x)

basemodel.trainable = False
model = Model(inputs=inputs, outputs=y)
model.summary()

trainGenerator = ImageDataGenerator(rescale=1./255, horizontal_flip = True, vertical_flip = True, rotation_range = 359)
testGenerator = ImageDataGenerator(rescale=1./255, horizontal_flip = True, vertical_flip = True, rotation_range = 359)

traindata = trainGenerator.flow_from_directory("data/train", target_size=(100, 100), batch_size=64, class_mode='binary')

validationdata = testGenerator.flow_from_directory("data/test", target_size=(100, 100), batch_size=64, class_mode='binary')

checkpointer = ModelCheckpoint(filepath = "weight.hdf5", save_best_only=True, save_weights_only=True)

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['acc'])

#model.load_weights("weight.hdf5")

model.fit_generator(traindata,
      steps_per_epoch=98,
      epochs=100,
      validation_data=validationdata,
      validation_steps=10,
	  callbacks=[checkpointer])















