"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. 
Specify the configuration settings at the beginning according to your 
problem.
This script was written for TensorFlow 1.0 and come with a blog post 
you can find here:
  
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert 
contact: f.kratzert(at)gmail.com
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Learning params
learning_rate = 0.0001
num_epochs = 10000
train_batch_size = 60000
test_batch_size = 10000

# Network params
dropout_rate = 1
num_classes = 10
train_layers = ['fc4', 'fc3', \
                'conv2', 'conv1']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./record/tfrecord"
checkpoint_path = "./record/tfrecord"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): 
    os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc4

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
  tf.summary.histogram(var.name, var)
  
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)
  

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

with tf.name_scope('test_metric'):
    test_accuracy = tf.placeholder(tf.float32, [])
    test_loss = tf.placeholder(tf.float32, [])
ts1 = tf.summary.scalar('test_accuracy', test_accuracy)
ts2 = tf.summary.scalar('test_loss', test_loss)

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator("TRAIN", shuffle = True)
val_generator = ImageDataGenerator("TEST", shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / train_batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / test_batch_size).astype(np.int16)

# Start Tensorflow session
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
 
  # Initialize all variables
  #sess.run(tf.global_variables_initializer())
  
  saver.restore(sess, 'record/tfrecord/model_epoch4612.ckpt')

  # Add the model graph to TensorBoard
  #writer.add_graph(sess.graph)
  
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(num_epochs):
        epoch += 4612
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step <= train_batches_per_epoch:
            
            print "batch: {}, step: {}".format(epoch+1, step)
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(train_batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys, 
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                print "displaying..."
                s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                        y: batch_ys, 
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
            step += 1
            
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_ls = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(test_batch_size)
            acc, ls = sess.run([accuracy, loss], feed_dict={x: batch_tx, 
                                                            y: batch_ty, 
                                                            keep_prob: 1.})
            test_acc += acc
            test_ls += ls
            test_count += 1
        test_acc /= test_count
        test_ls /= test_count
        
        s1 = sess.run(ts1, feed_dict={test_accuracy: np.float32(test_acc)})
        s2 = sess.run(ts2, feed_dict={test_loss: np.float32(test_ls)})
        writer.add_summary(s1, (epoch+1)*train_batches_per_epoch)
        writer.add_summary(s2, (epoch+1)*train_batches_per_epoch)

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        
