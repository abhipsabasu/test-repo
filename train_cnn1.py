'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Patel Bhargavkumar Kanubhai
Roll No.: 16CS60R36

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf


cnnWeightsPath = "cnn/cnnWeight"

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def train(trainX, trainY):
    '''
    Complete this function.
    '''
    trainX = trainX.reshape([-1,784])
    oneHotTrainY = np.array([ np.array([0 for i in range(10)]) for y in range(len(trainY)) ])
    for i in range(len(trainY)):
        oneHotTrainY[i][trainY[i]] = 1

    print(trainX.shape, oneHotTrainY.shape) 


    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    #sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    saver.restore(sess, cnnWeightsPath)

    batch_size = 1000
    num_batch = 60000/batch_size    

    for _ in range(60000):

        batch_num = _%num_batch

        btrainX = trainX[batch_num*batch_size:(batch_num+1)*batch_size]
        boneHotTrainY = oneHotTrainY[batch_num*batch_size:(batch_num+1)*batch_size]

        sess.run(train_step, feed_dict={x: btrainX, y_: boneHotTrainY,keep_prob:0.5})
        print(_,sess.run(accuracy, feed_dict={x: btrainX, y_: boneHotTrainY,keep_prob:0.5}))

        if _%10==0:
            save_path = saver.save(sess, cnnWeightsPath)

    save_path = saver.save(sess, cnnWeightsPath)
    # print("Model saved in file: %s" % save_path)
    


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    testX = testX.reshape([-1,784])

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, cnnWeightsPath)
    # print("Model restored.")

    predictions = sess.run(tf.argmax(y_conv,1), feed_dict={x: testX,keep_prob:1.0})

    return predictions
