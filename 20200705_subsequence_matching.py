import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization

from tensorflow.keras.optimizers import SGD

def letter_to_index(letter):
	_alphabet = 'atgc'
	return next(((i + 1) / 4 for i, _letter in enumerate(_alphabet) if _letter == letter), None)

mypath = '/Users/ethanmoyer/Desktop/Coursework/1920_spring/VIP/subsequence-matching/data/ref_sequences/'

data = []

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '20' in f]

for file in files[:20]:
	data.append(pd.read_csv(mypath + file))

for entry_data in data:
	entry_data['Subsequence'] = entry_data['Subsequence'].apply(lambda x: [float(letter_to_index(elem)) for elem in x])

	a = np.array(entry_data['Subsequence'].tolist())

	a = a.reshape((1, a.shape[0], 20))

	b = np.array(entry_data['Contains'].tolist())

	if 'features' not in locals():
		features = a
		outputs = b
	else:
		features = np.vstack((features, a))
		outputs = np.vstack((outputs, b))

time_steps = features.shape[1]

n_features = features.shape[2]

# define model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(time_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=256, kernel_size=3, activation='linear',
			padding='same', strides=3))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=512, kernel_size=3, activation='linear',
			padding='same', strides=3))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=2048, kernel_size=3, activation='linear',
			padding='same', strides=3))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
#model.add(Dense(50, activation='relu'))
model.add(Dense(len(outputs[0])))

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

model.summary()

# fit model
model.fit(features, outputs, epochs = 100, batch_size = 64, verbose=1)
# demonstrate prediction
#x_input = array([70, 80, 90])
#x_input = x_input.reshape((1, n_steps, n_features))
ypred = model.predict(features[1:2], verbose=0)[0][:20]
yact = outputs[1][:20]
print(ypred)
print(yact)


#print(yhat)