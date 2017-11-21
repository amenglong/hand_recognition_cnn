import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from scipy import ndimage
from cnn_utils import predict, load_dataset

## PUT IMAGE NAME ## 
my_image = "Al_hand.jpg"
## END IMAGE NAME ##

def initialize_parameters():    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
        
    W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

# Open trained model
parameters = initialize_parameters()

saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, "variables/save_variables.ckpt")
	parameters = sess.run(parameters)

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64, 64, 3))
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
plt.show()