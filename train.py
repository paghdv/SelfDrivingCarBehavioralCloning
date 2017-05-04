import csv
import cv2
import numpy as np
lines = []

#Load file names and steering angles
print('Starting training process')
with open('driving_training_2/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

print('Image paths loaded from csv, starting image loading...')
images = [];
measurements = [];

#Load images and augment data
for line in lines:
	for i in range(0,3):#Used to load center, left and right
		source_path = line[i];
		path_parts = source_path.split('/')
		filename = path_parts[-1]
		folder = path_parts[len(path_parts)-2]
		current_path = 'driving_training_2/'+folder+'/' + filename
		image = cv2.imread(current_path)
		if image is None:
			continue
		measurement = float(line[3])
		if (filename[0:4]=='left'):
			measurement = measurement + 0.2;
		if (filename[0:5]=='right'):			
        	        measurement = measurement - 0.2;
		images.append(image)
		measurements.append(measurement)
		images.append(cv2.flip(image,1))
		measurements.append(-measurement)

print('Images loaded, starting training...')


#Initializing keras and importing libraries/modules

X_train = np.array(images)
y_train = np.array(measurements)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((25,25),(0,0))))

# Nvidias model
model.add(Convolution2D(3 ,5 ,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(24,5 ,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5 ,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,3 ,3,activation='relu'))
model.add(Convolution2D(64,3 ,3,activation='relu'))
#model.add(Convolution2D(64,1,18,activation='relu'))
model.add(Flatten())
model.add(Dense(1164, 	activation='relu'))
model.add(Dense(100, 	activation='relu'))
model.add(Dense(50, 	activation='relu'))
model.add(Dense(10, 	activation='relu'))
model.add(Dense(1))

print('===============================')
print('Training started with %d images',len(images))
print('===============================')

# Training
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.15,shuffle=True,epochs=15,verbose=1)
# Saving
model.save('nvidia_1dconv_v7_15it.h5')

print('Training done with %d images',len(images))

exit()

