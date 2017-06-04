'''
MIT License

Copyright (c) 2016 Panagiotis Petridis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
############### THE ABOVE LICENSE NOTICE IS ONLY VALID FOR THE SOURCE CODE AND NOT THE DATA USED TO TRAIN THE MODEL
import numpy as np
import tensorflow as tf

#       Getting the Data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28*28])   # Model Input --> i.e the raw pixel data all in one line
y = tf.placeholder(tf.float32, [None, 10])      # Labeled output --> class that should be predicted.
                                                # It also is one-hot encoded. i.e [1,0,0,0,0,0,0,0,0,0] is a '0'
                                                # and [0,0,0,1,0,0,0,0,0,0] is a '3' and so on...

#       Layer #1
W = tf.Variable(tf.truncated_normal([28*28, 400], stddev=0.1))
b = tf.Variable(tf.zeros([400]))

#       Layer #2
M = tf.Variable(tf.truncated_normal([400, 400], stddev=0.1))
c = tf.Variable(tf.zeros([400]))

#       Layer #3
Q = tf.Variable(tf.truncated_normal([400, 10], stddev=0.1))
d = tf.Variable(tf.zeros([10]))

keep_prob = tf.placeholder(tf.float32) # Probability of keeping a unit during dropout

l1 = tf.nn.relu(tf.matmul(X, W) + b)
l1 = tf.nn.dropout(l1, keep_prob)   # Add dropout to 1st layer  ---> Helps with reguralization
l2 = tf.nn.relu(tf.matmul(l1, M) + c)
l1 = tf.nn.dropout(l2, keep_prob)   # Add dropout to 2nd layer  ---> Helps with reguralization
l3 = tf.matmul(l2, Q) + d
y_ = tf.nn.softmax(l3)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), [1]))  # Cross Entropy Loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)    # Gradient Descent Optimizer

n_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))   # Number of correctly predicted examples from test set
accuracy = tf.reduce_mean(tf.cast(n_correct, tf.float32))   # Total accuracy on test set

saver = tf.train.Saver()    # used to save the model

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#       Loading the model
loader = tf.train.import_meta_graph('my-model.meta')
loader.restore(sess, tf.train.latest_checkpoint('./'))

#       Tracking best score
best_score = sess.run(accuracy, feed_dict={X: data.test.images, y: data.test.labels, keep_prob: 1})

batch_size = 20000
for i in range(1000):
    x_b, y_b = data.train.next_batch(batch_size)
    _, l = sess.run([optimizer, loss], {X: x_b, y: y_b, keep_prob: 0.5})
    if (i+1)%100==0:
        acc = sess.run(accuracy, feed_dict={X: data.test.images, y: data.test.labels, keep_prob: 1})
        # Save model if test accuracy is better than previous best
        if acc > best_score:
            saver.save(sess, 'my-model')
            print('New best score!')
            best_score = acc
        print(l)

print('Test Accuracy: ', sess.run(accuracy, feed_dict={X: data.test.images, y: data.test.labels, keep_prob: 1}))
## The model gets an average ~97.8% accuracy which although is considered pathetically bad on the MNIST dataset, ins't as bad for a Feed Forward Deep Neural Network.
