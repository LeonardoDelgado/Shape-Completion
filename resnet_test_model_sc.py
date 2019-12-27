from keras.models import load_model
import numpy as np
from utilities_for_data import split_views
from utilities_for_data import views_to_boxel
from utilities_for_data import boxel_to_cloud
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


datasets_directory = '/media/leo/Datasets/' 
PATH = 'models laptop2 2019 08 30/'
NAME = 'model-152-47.010418.h5'
model = load_model(PATH + NAME)

X = np.load(datasets_directory + 'samples_18_14.npy')
Y = np.load(datasets_directory + 'labels_18_14.npy')
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#viewx_1, viewx_2, viewx_3, viewx_4, viewx_5, viewx_6 = split_views(X_train)
#viewy_1, viewy_2, viewy_3, viewy_4, viewy_5, viewy_6 = split_views(y_train)
#del(X_train,y_train)

viewx_1_val, viewx_2_val, viewx_3_val, viewx_4_val, viewx_5_val, viewx_6_val = split_views(X)
#viewy_1_val, viewy_2_val, viewy_3_val, viewy_4_val, viewy_5_val, viewy_6_val = split_views(y_test)



decoded_imgs =  model.predict([viewx_1_val, viewx_2_val, viewx_3_val, viewx_4_val, viewx_5_val, viewx_6_val])
decoded_imgs = np.concatenate(decoded_imgs,axis=3)
print(decoded_imgs.shape)


np.save('/media/leo/Datasets/results_of_proposal.npy',decoded_imgs)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
n = 6
img = 10
plt.figure(figsize = (18,6))
for i in range(n):
	ax = plt.subplot(3, n, i+1)
	plt.imshow(X[img,:,:,i].reshape(32,32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + n+1)
	plt.imshow(decoded_imgs[img,:,:,i].reshape(32,32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + 2*n+1)
	plt.imshow(Y[img,:,:,i].reshape(32,32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()

voxel_cube = views_to_boxel(decoded_imgs[img,:,:,:],32)
x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.get_zaxis().set_visible(False)
	
plt.show()
    
