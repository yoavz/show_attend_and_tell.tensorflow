import os
import sys
import numpy as np
import skimage
import skimage.io

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model_tensorflow import Caption_Generator

CONCAT_LENGTH = 1

class MNISTCaptionGenerator(Caption_Generator):

    def _init_conv_net(self):
        """
        Defines the parameters of convolution to go from
        image -> context vectors. Image size starts at
        28 x (28 * CONCAT_LENGTH). 
        Context vector is (49 * CONCAT_LENGTH) x 128
        """

        with tf.variable_scope("conv1"): 
            self.conv1_W_1 = self.init_custom_weight([3, 3, 1, 64], name="W_1")
            self.conv1_b_1 = self.init_bias(64, name="b_1")
            self.conv1_W_2 = self.init_custom_weight([3, 3, 64, 64], name="W_2")
            self.conv1_b_2 = self.init_bias(64, name="b_2")

        with tf.variable_scope("conv2"): 
            self.conv2_W_1 = self.init_custom_weight([3, 3, 64, 128], name="W_1")
            self.conv2_b_1 = self.init_bias(128, name="b_1")
            self.conv2_W_2 = self.init_custom_weight([3, 3, 128, 128], name="W_2")
            self.conv2_b_2 = self.init_bias(128, name="b_2")

    def build_conv_net(self, images):
        """
        Build the convolution section of the graph (image -> context)
        """
        
        # reshape images into 4 rank for convolution operation: [ batch, 28, 280, 1]
        images_shaped = tf.expand_dims(images, -1)

        # 28 x 280
        conv1_1 = self.conv_layer(images_shaped, self.conv1_W_1, self.conv1_b_1)
        conv1_2 = self.conv_layer(conv1_1, self.conv1_W_2, self.conv1_b_2)
        pool1 = self.max_pool(conv1_2)

        # 14 x 140 
        assert pool1.get_shape().as_list()[1:] == [14, 14*CONCAT_LENGTH, 64]

        conv2_1 = self.conv_layer(pool1, self.conv2_W_1, self.conv2_b_1)
        conv2_2 = self.conv_layer(conv2_1, self.conv2_W_2, self.conv2_b_2)
        pool2 = self.max_pool(conv2_2)

        # 7 x 70
        assert pool2.get_shape().as_list()[1:] == [7, 7*CONCAT_LENGTH, 128]

        return tf.reshape(pool2, [-1, 49*CONCAT_LENGTH, 128])

##### Parameters ######
n_epochs=1000
save_every=5 # save every 5 epochs
batch_size=1000/CONCAT_LENGTH
# technically we don't need a word embedding for MNIST labels,
# but use one anyways to test the model
dim_embed=10
dim_ctx=128
dim_hidden=256
img_shape=[28,28*CONCAT_LENGTH]
model_path = 'models'
learning_rate=0.001
#############################

def horizontally_stack(a, axis):
    return np.squeeze(np.concatenate(np.split(a, CONCAT_LENGTH), axis=axis))

def train():
    mnist_data = input_data.read_data_sets("data/MNIST/", one_hot=True)
    num_train = mnist_data.train.images.shape[0]
    mnist_train_images = np.reshape(mnist_data.train.images, (num_train, 28, 28))
    mnist_train_labels = np.reshape(mnist_data.train.labels, (num_train, 10))
    mnist_train_labels = np.nonzero(mnist_train_labels)[1] # one hot to integer
    mnist_train_labels = np.reshape(mnist_train_labels, (num_train/CONCAT_LENGTH, CONCAT_LENGTH))

    stacked = np.stack([horizontally_stack(m, 2) for m in np.split(mnist_train_images, num_train/CONCAT_LENGTH, axis=0)])

    # skimage.io.imsave("train_image.png", stacked[8, :, :])
    # print mnist_train_labels[8, :]

    sess = tf.InteractiveSession()

    caption_generator = MNISTCaptionGenerator(
        n_words=10, # 10 possible words
        dim_embed=dim_embed,
        dim_ctx=dim_ctx,
        dim_hidden=dim_hidden,
        n_lstm_steps=CONCAT_LENGTH, 
        batch_size=batch_size,
        img_shape=img_shape,
        bias_init_vector=None)

    loss, images, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=100)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()

    batches_per_epoch = mnist_train_labels.shape[0]/batch_size

    print "Training on {} images concatenated together horizontally".format(CONCAT_LENGTH)
    print "{} total train images".format(mnist_train_images.shape[0])
    for epoch in range(n_epochs):
        print "Epoch: {}".format(epoch)
        for batch_num in range(batches_per_epoch):

            current_images = stacked[batch_num*batch_size:(batch_num+1)*batch_size, :, :]
            current_sentence = mnist_train_labels[batch_num*batch_size:(batch_num+1)*batch_size, :]
            current_mask = np.ones((batch_size, mnist_train_labels.shape[1]))

            _, loss_value = sess.run([train_op, loss], feed_dict={
                images: current_images,
                sentence: current_sentence,
                mask: current_mask
            })
        
            print "Current Cost: {} (batch {}/{})".format(loss_value, batch_num, batches_per_epoch)

        if epoch % save_every == 0:
            saver.save(sess, os.path.join(model_path, 'mnist'), global_step=epoch/save_every)

def test(model_name="mnist-10"):
    # TODO(yoavz): fix with CONCAT_LENGTH
    mnist_data = input_data.read_data_sets("data/MNIST", one_hot=True)
    num_test = mnist_data.test.images.shape[0]
    mnist_test_images = np.reshape(mnist_data.test.images, (num_test, 28, 28))
    mnist_test_labels = np.reshape(mnist_data.test.labels, (num_test, 10))
    mnist_test_labels = np.nonzero(mnist_test_labels)[1] # one hot to integer
    mnist_test_labels = np.reshape(mnist_test_labels, (num_test/CONCAT_LENGTH, CONCAT_LENGTH))

    stacked = np.stack([horizontally_stack(m, 2) for m in np.split(mnist_test_images, num_test/CONCAT_LENGTH, axis=0)])

    skimage.io.imsave("test_image.png", stacked[0, :, :])
    print "Test image saved to test_image.png"

    sess = tf.InteractiveSession()

    caption_generator = MNISTCaptionGenerator(
        n_words=10, # 10 possible words
        dim_embed=dim_embed,
        dim_ctx=dim_ctx,
        dim_hidden=dim_hidden,
        n_lstm_steps=CONCAT_LENGTH
        batch_size=batch_size,
        img_shape=img_shape,
        bias_init_vector=None)

    images, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=CONCAT_LENGTH)
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(model_path, model_name))

    generated_words = sess.run(generated_words, feed_dict = { images: np.expand_dims(stacked[0, :, :], axis=0) })
    print generated_words

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test(sys.argv[1])
    else:
        train()
