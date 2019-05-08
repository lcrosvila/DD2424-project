import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train
x_test = x_test
#y_train = y_train[0:3]

# Number of training examples
n_train = len(x_train) 

# Number of testing examples.
n_test = len(x_test) 

# What's the shape of an traffic sign image?
image_shape = x_train[0].shape 

# How many unique classes/labels there are in the dataset.
#n_classes = len(set(y_train)) #set() returns unordered collection of unique elements
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print(type(x_train[0]))

def normalize(img):
    img_array = np.asarray(img)
    normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    return normalized

x_train = normalize(x_train)
x_test = normalize(x_test)

# Store layers weight & bias
weights = {
    # first layer
    'wc1_1': tf.Variable(tf.random_normal([5, 5, 3, 192])*tf.math.sqrt(2./(5*5*192)), name='wc1_1'),
    'wc1_2': tf.Variable(tf.random_normal([1, 1, 192, 160])*tf.math.sqrt(2./(160)), name='wc1_2'),
    'wc1_3': tf.Variable(tf.random_normal([1, 1, 160, 96])*tf.math.sqrt(2./(96)), name='wc1_3'),
    # second layer
    'wc2_1': tf.Variable(tf.random_normal([5,5,96,192])*tf.math.sqrt(2./(5*5*192)), name='wc2_1'),
    'wc2_2': tf.Variable(tf.random_normal([1,1,192,192])*tf.math.sqrt(2./(192)), name='wc2_2'),
    'wc2_3': tf.Variable(tf.random_normal([1,1,192,192])*tf.math.sqrt(2./(192)), name='wc2_3'),
    # third layer
    'wc3_1': tf.Variable(tf.random_normal( [3,3,192,192])*tf.math.sqrt(2./(3*3*192)), name='wc3_1'),
    'wc3_2': tf.Variable(tf.random_normal( [1,1,192,192])*tf.math.sqrt(2./(192)), name='wc3_2'),
    'wc3_3': tf.Variable(tf.random_normal( [1,1,192,192])*tf.math.sqrt(2./(192)), name='wc3_3'),
    # fourth layer
    'wc4_1': tf.Variable(tf.random_normal( [3,3,192,192])*tf.math.sqrt(2./(3*3*192)), name='wc4_1'),
    'wc4_2': tf.Variable(tf.random_normal( [1,1,192,192])*tf.math.sqrt(2./(192)), name='wc4_2'),
    'wc4_3': tf.Variable(tf.random_normal( [1,1,192,192])*tf.math.sqrt(2./(192)), name='wc4_3')
}

biases = {
    'bc1_1': tf.Variable(tf.zeros([192]), name='bc1_1'),
    'bc1_2': tf.Variable(tf.zeros([160]), name='bc1_2'),
    'bc1_3': tf.Variable(tf.zeros([96]),  name='bc1_3'),
    'bc2_1': tf.Variable(tf.zeros([192]), name='bc2_1'),
    'bc2_2': tf.Variable(tf.zeros([192]), name='bc2_2'),
    'bc2_3': tf.Variable(tf.zeros([192]), name='bc2_3'),
    'bc3_1': tf.Variable(tf.zeros([192]), name='bc3_1'),
    'bc3_2': tf.Variable(tf.zeros([192]), name='bc3_2'),
    'bc3_3': tf.Variable(tf.zeros([192]), name='bc3_3'),
    'bc4_1': tf.Variable(tf.zeros([192]), name='bc4_1'),
    'bc4_2': tf.Variable(tf.zeros([192]), name='bc4_2'),
    'bc4_3': tf.Variable(tf.zeros([192]), name='bc4_3')
}

def conv_layer(x, x_shape, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.contrib.layers.batch_norm(x)
    return tf.nn.relu(x)

def max_pooling (x,kernel_size=3,stride=2): 
    return tf.nn.max_pool (x, ksize = [1, kernel_size, kernel_size, 1], strides = [1, stride, stride, 1], padding = 'SAME'  ) 

def avg_pooling(x,kernel_size=3,stride=2):
    return tf.nn.avg_pool (x, ksize = [1, kernel_size, kernel_size, 1], strides = [1, stride, stride, 1], padding = 'SAME'  )

def global_avg_pooling(x,kernel_size=3,stride=2):
    return tf.reduce_mean(x, axis=[1,2])

def ConvNet (x): 
    #print('x', x.shape)
    block_1_1 = conv_layer(x, [5,5,3,192], weights['wc1_1'], biases['bc1_1']  ) 
    #print('block_1_1', block_1_1.shape)
    block_1_2 = conv_layer(block_1_1, [1,1,192,160], weights['wc1_2'], biases['bc1_2']  )
    #print('block_1_2', block_1_2.shape)
    block_1_3 = conv_layer(block_1_2, [1,1,160,96] , weights['wc1_3'], biases['bc1_3'] )
    #print('block_1_3', block_1_3.shape)
    block_1_pool = max_pooling (block_1_3)
    #print('block_1_pool', block_1_pool.shape)

    block_2_1 = conv_layer(block_1_pool, [5,5,96,192], weights['wc2_1'], biases['bc2_1']  ) 
    #print('block_2_1', block_2_1.shape)
    block_2_2 = conv_layer(block_2_1, [1,1,192,192], weights['wc2_2'], biases['bc2_2']  )
    #print('block_2_2', block_2_2.shape)
    block_2_3 = conv_layer(block_2_2, [1,1,192,192], weights['wc2_3'], biases['bc2_3'] )
    #print('block_2_3', block_2_3.shape)
    block_2_pool = avg_pooling (block_2_3)
    #print('block_2_pool', block_2_pool.shape)

    block_3_1 = conv_layer(block_2_pool, [3,3,192,192], weights['wc3_1'], biases['bc3_1']  ) 
    #print('block_3_1', block_3_1.shape)
    block_3_2 = conv_layer(block_3_1, [1,1,192,192], weights['wc3_2'], biases['bc3_2']  )
    #print('block_3_2', block_3_2.shape)
    block_3_3 = conv_layer(block_3_2, [1,1,192,192], weights['wc3_3'], biases['bc3_3'] )
    #print('block_3_3', block_3_3.shape)
    block_3_pool = avg_pooling (block_3_3)
    #print('block_3_pool', block_3_pool.shape)

    block_4_1 = conv_layer(block_3_pool, [3,3,192,192], weights['wc4_1'], biases['bc4_1']  ) 
    #print('block_3_1', block_3_1.shape)
    block_4_2 = conv_layer(block_4_1, [1,1,192,192], weights['wc4_2'], biases['bc4_2']  )
    #print('block_3_2', block_3_2.shape)
    block_4_3 = conv_layer(block_4_2, [1,1,192,192], weights['wc4_3'], biases['bc4_3'] )
    #print('block_3_3', block_3_3.shape)
    block_4_pool = global_avg_pooling (block_4_3)
  
  
    fully_layer = tf.contrib.slim.fully_connected(block_4_pool, 4, activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm)
    fully_layer = tf.nn.softmax(fully_layer)

    return fully_layer


def get_one_hot(x,K):
    one_hot=np.zeros((np.shape(x)[0],K))
    one_hot[:,0]=1
    for j in range(K-1):
        labels=np.zeros((np.shape(x)[0],K))
        labels[:,j+1]=1
        one_hot=np.concatenate((one_hot,labels))
    return one_hot

from sklearn.utils import shuffle
from PIL import Image
from sklearn.utils import shuffle

EPOCHS = 100
#EPOCHS = 4
BATCH_SIZE = 128
lr = tf.Variable(0.1) # learning rate
MOMENTUM = 0.9
DECAY = 5e-4 # WE NEED TO ADD THIS AS SOME KIND OF REGULARIZATION
K=4 

num_examples = len(x_train)
num_test = len(x_test)
x = tf.placeholder(tf.float32, shape=(None,)+image_shape)
x1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE,)+image_shape)
y = tf.placeholder(tf.int32)
Y = tf.placeholder(tf.float32, shape=(None,K))
Y2 = tf.placeholder(tf.float32, shape=(None,K))
N=tf.placeholder(tf.float32)
K_L=tf.placeholder(tf.int32)

x_rot =  tf.image.rot90(x, y)

F = ConvNet(x)

#one_hot2=get_one_hot2(x1,K_L)
regularizers = tf.nn.l2_loss(weights['wc1_1']) + tf.nn.l2_loss(weights['wc1_2']) + tf.nn.l2_loss(weights['wc1_3']) +\
               tf.nn.l2_loss(weights['wc2_1']) + tf.nn.l2_loss(weights['wc2_2']) + tf.nn.l2_loss(weights['wc2_3']) +\
               tf.nn.l2_loss(weights['wc3_1']) + tf.nn.l2_loss(weights['wc3_2']) + tf.nn.l2_loss(weights['wc3_3']) +\
               tf.nn.l2_loss(weights['wc4_1']) + tf.nn.l2_loss(weights['wc4_2']) + tf.nn.l2_loss(weights['wc4_3'])

loss=tf.math.divide(-tf.reduce_sum(tf.math.log(tf.reduce_sum(F*Y,1)),0),(K*N)) + DECAY * regularizers

#loss = tf.math.divide(-tf.reduce_sum(tf.math.log(tf.reduce_sum(F*Y,1)),0),(K*N))

acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(Y, 1),predictions=tf.argmax(F, 1))

acc1=tf.contrib.metrics.accuracy(predictions=tf.argmax(F, 1),labels=tf.argmax(Y, 1))

acc2=tf.contrib.metrics.accuracy(predictions=tf.argmax(Y2, 1),labels=tf.argmax(Y, 1))

optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=MOMENTUM, use_nesterov=True)

train = optimizer.minimize(loss)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: 
    sess.run(tf.global_variables_initializer())
    print("Training...")
    for i in range(EPOCHS):
        #sess.run(tf.local_variables_initializer())
        print('epoch: ', i+1)
        if i == 29 or i == 59 or i == 79:
            lr = lr/5
        x_train = shuffle(x_train)
        for index in range(0, num_examples, BATCH_SIZE):
            end = index + BATCH_SIZE
            batch_x = x_train[index:end]
            labels=get_one_hot(batch_x,K)
            batch_x_rot=sess.run ( x_rot, feed_dict={x:batch_x,y:0})
            for j in range(K-1):
                batch_x_rot=np.concatenate((batch_x_rot,sess.run ( x_rot, feed_dict={x:batch_x, y:(j+1)})))
        
            if index==0:
                loss_val = sess.run ( loss, feed_dict={Y:labels,x:batch_x_rot,N:len(batch_x_rot)})
                print("loss",loss_val)
            sess.run(train, feed_dict={Y:labels,x:batch_x_rot,N:len(batch_x_rot)})
    
        x_test_rot=sess.run ( x_rot, feed_dict={x:x_test,y:0})
        labels_test=get_one_hot(x_test,K)
        for j in range(K-1):
            x_test_rot=np.concatenate((x_test_rot,sess.run ( x_rot, feed_dict={x:x_test, y:(j+1)})))  
    
        
        batch_size_test =int(num_test/4)
        accuracy_test=[]
        for index in range(0, 4*num_test, batch_size_test):
            end = index + batch_size_test
            batch_test = x_test_rot[index:end]
            labels_test_batch=labels_test[index:end]
            accuracy_test=np.concatenate((accuracy_test,[sess.run ( acc1, feed_dict={x:batch_test, Y:labels_test_batch})]))     
        accuracy_test=np.sum(accuracy_test)/len(accuracy_test)
        print("accuracy :",accuracy_test)
