import os, sys
import math
import cPickle
import numpy as np
import skimage
import skimage.io
from collections import defaultdict

import tqdm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model_tensorflow import Caption_Generator

##### Data Location ######
# train_formula_images_path = 'im2latex-dataset/train_formula_images'
# train_metadata_path = 'im2latex-dataset/train_im2latex.lst'
# train_formulas_path = 'im2latex-dataset/train_im2latex_formulas.lst'
train_formula_images_path = 'im2latex-dataset/formula_images'
train_metadata_path = 'im2latex-dataset/im2latex.lst'
train_formulas_path = 'im2latex-dataset/im2latex_formulas.lst'
test_formula_images_path = 'im2latex-dataset/test_formula_images'
test_metadata_path = 'im2latex-dataset/test_im2latex.lst'
test_formulas_path = 'im2latex-dataset/test_im2latex_formulas.lst'

##### Model Parameters ######
n_epochs=100
save_every=1 
batch_size=100
dim_ctx=512
dim_hidden=256
img_shape=[64, 512]
max_lstm_steps=81 # 80 length + 1 for ending token
model_path = 'models/im2latex'
learning_rate=0.001

class IM2LATEXCaptionGenerator(Caption_Generator):

    def _init_conv_net(self):
        """
        Defines the parameters of convolution to go from
        image -> context vectors. Image size starts at 64 x 512
        Context vectors are 128 x 512.

        Copies VGG16 architecture.
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

        with tf.variable_scope("conv3"): 
            self.conv3_W_1 = self.init_custom_weight([3, 3, 128, 256], name="W_1")
            self.conv3_b_1 = self.init_bias(256, name="b_1")
            self.conv3_W_2 = self.init_custom_weight([3, 3, 256, 256], name="W_2")
            self.conv3_b_2 = self.init_bias(256, name="b_2")
            self.conv3_W_3 = self.init_custom_weight([3, 3, 256, 256], name="W_3")
            self.conv3_b_3 = self.init_bias(256, name="b_3")
         
        with tf.variable_scope("conv4"): 
            self.conv4_W_1 = self.init_custom_weight([3, 3, 256, 512], name="W_1")
            self.conv4_b_1 = self.init_bias(512, name="b_1")
            self.conv4_W_2 = self.init_custom_weight([3, 3, 512, 512], name="W_2")
            self.conv4_b_2 = self.init_bias(512, name="b_2")
            self.conv4_W_3 = self.init_custom_weight([3, 3, 512, 512], name="W_3")
            self.conv4_b_3 = self.init_bias(512, name="b_3")

        # with tf.variable_scope("conv5"): 
        #     self.conv5_W_1 = self.init_custom_weight([3, 3, 512, 512], name="W_1")
        #     self.conv5_b_1 = self.init_bias(512, name="b_1")
        #     self.conv5_W_2 = self.init_custom_weight([3, 3, 512, 512], name="W_2")
        #     self.conv5_b_2 = self.init_bias(512, name="b_2")
        #     self.conv5_W_3 = self.init_custom_weight([3, 3, 512, 512], name="W_3")
        #     self.conv5_b_3 = self.init_bias(512, name="b_3")


    def build_conv_net(self, images):
        """
        Build the convolution section of the graph (image -> context)
        """

        # reshape images into 4 rank for convolution operation: [ batch, h, w, 1]
        images_shaped = tf.expand_dims(images, -1)

        # 128 x 1024
        conv1_1 = self.conv_layer(images_shaped, self.conv1_W_1, self.conv1_b_1)
        conv1_2 = self.conv_layer(conv1_1, self.conv1_W_2, self.conv1_b_2)
        pool1 = self.max_pool(conv1_2)

        # 64 x 512
        # assert pool1.get_shape().as_list()[1:] == [64, 512, 64]
        # 32 x 256
        assert pool1.get_shape().as_list()[1:] == [32, 256, 64]

        conv2_1 = self.conv_layer(pool1, self.conv2_W_1, self.conv2_b_1)
        conv2_2 = self.conv_layer(conv2_1, self.conv2_W_2, self.conv2_b_2)
        pool2 = self.max_pool(conv2_2)

        # 32 x 256
        # assert pool2.get_shape().as_list()[1:] == [32, 256, 128]
        # 16 x 128
        assert pool2.get_shape().as_list()[1:] == [16, 128, 128]

        conv3_1 = self.conv_layer(pool2, self.conv3_W_1, self.conv3_b_1)
        conv3_2 = self.conv_layer(conv3_1, self.conv3_W_2, self.conv3_b_2)
        conv3_3 = self.conv_layer(conv3_2, self.conv3_W_3, self.conv3_b_3)
        pool3 = self.max_pool(conv3_3)

        # 16 x 128
        assert pool3.get_shape().as_list()[1:] == [8, 64, 256]

        conv4_1 = self.conv_layer(pool3, self.conv4_W_1, self.conv4_b_1)
        conv4_2 = self.conv_layer(conv4_1, self.conv4_W_2, self.conv4_b_2)
        conv4_3 = self.conv_layer(conv4_2, self.conv4_W_3, self.conv4_b_3)
        pool4 = self.max_pool(conv4_3)

        # 8 x 64
        assert pool4.get_shape().as_list()[1:] == [4, 32, 512]

        # conv5_1 = self.conv_layer(pool4, self.conv5_W_1, self.conv5_b_1)
        # conv5_2 = self.conv_layer(conv5_1, self.conv5_W_2, self.conv5_b_2)
        # conv5_3 = self.conv_layer(conv5_2, self.conv5_W_3, self.conv5_b_3)
        # pool5 = self.max_pool(conv5_3)
        #
        # # 4 x 32 (x 512 filters)
        # assert pool5.get_shape().as_list()[1:] == [4, 32, 512]

        return tf.reshape(pool4, [-1, 128, 512])

def load_data(formula_images_path, formulas_path, metadata_path):

    alphabet = set()
    formulas = []

    # read all formulas and collect alphabet set
    with open(formulas_path, 'r') as f:
        for line in f:
            formulas.append(line)
            alphabet |= set(line)

    # first index is reserved for "end" character
    char_to_idx = { '\0': 0 }
    idx = 1
    for c in alphabet:
        char_to_idx[c] = idx
        idx += 1

    # open metadata to get the path of all images
    data = []
    with open(metadata_path, 'r') as f:
        for line in f:
            idx, image_name = line.split()[:2]
            idx = int(idx)
            if os.path.isfile(os.path.join(formula_images_path, "{}.png".format(image_name))):
                data.append((formulas[idx], image_name))
            else:
                print "{}.png not found, skipping".format(image_name)

    print "Finished loading {} images".format(len(data))

    return data, char_to_idx

def load_images(image_names, formula_images_path):

    image_data = np.zeros((len(image_names), img_shape[0], img_shape[1]))

    for i, name in enumerate(image_names):
        try:
            img = skimage.io.imread(os.path.join(formula_images_path, 
                "{}.png".format(name))) 
        except Exception as e:
            print e
            return None

        if img.shape[0] != img_shape[0] or img.shape[1] != img_shape[1]:
            print "Found inconsistent size: {}".format(img.shape)
            return None

        img = img / 255.0
        image_data[i, :, :] = img

    return image_data

def train():

    data, char_to_idx = load_data(train_formula_images_path, 
                                  train_formulas_path, 
                                  train_metadata_path)
    N = len(data) # number of training examples
    n_words = len(char_to_idx)

    sess = tf.InteractiveSession()

    caption_generator = IM2LATEXCaptionGenerator(
        n_words=n_words, 
        dim_ctx=dim_ctx,
        dim_hidden=dim_hidden,
        n_lstm_steps=max_lstm_steps, 
        batch_size=batch_size,
        img_shape=img_shape,
        dropout=0.8)

    loss, images, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=100)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()

    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    with open(os.path.join(model_path, 'dictionary.pickle'), 'w') as f:
        cPickle.dump(char_to_idx, f)

    # WARNING: ignores the final remainder of batches for simplicity
    batches_per_epoch = N / batch_size
    
    for epoch in range(n_epochs):

        train_order = np.arange(N)
        np.random.shuffle(train_order)

        print "Epoch: {}".format(epoch)

        for batch_num in range(batches_per_epoch):

            batch_idxs = train_order[batch_num*batch_size:(batch_num+1)*batch_size]

            current_image_names = [data[i][1] for i in batch_idxs]
            current_images = load_images(current_image_names, train_formula_images_path)
            if current_images is None:
                print "Problem loading image in batch {}, skipping".format(batch_num)
                continue
            current_formulas = [data[i][0] for i in batch_idxs]

            # current_images = images_data[batch_num*batch_size:(batch_num+1)*batch_size, :, :]
            # current_formulas = formulas[batch_num*batch_size:(batch_num+1)*batch_size]
            
            current_labels = np.zeros((batch_size, max_lstm_steps), dtype=np.int32)
            current_mask = np.zeros((batch_size, max_lstm_steps), dtype=np.int32)
            for i, f in enumerate(current_formulas):
                # add 0 at the end for the terminating character
                current_labels[i, :len(f)+1] = [ char_to_idx[c] for c in f ] + [0]
                # set mask of the example to ones
                current_mask[i, :len(f)+1] = np.ones(len(f)+1, dtype=np.int32)
            
            
            # uncomment this to test that the formulas and images line up
            # loss_value = 0
            # rand = np.random.randint(0, batch_size)
            # print "{}: {}".format(batch_num, current_formulas[rand])
            # skimage.io.imsave("test/{}.png".format(batch_num), current_images[rand, :, :])

            _, loss_value = sess.run([train_op, loss], feed_dict={
                images: current_images,
                sentence: current_labels,
                mask: current_mask
            })

            print "Current Cost: {} (batch {}/{})".format(loss_value, batch_num, batches_per_epoch)

        if epoch % save_every == 0:
            saver.save(sess, os.path.join(model_path, 'im2latex'), global_step=epoch/save_every)

# def test():
#
#     data = load_data(test_formula_images_path,
#                      test_formulas_path,
#                      test_metadata_path)
#
#     with open(os.path.join(model_path, 'dictionary.pickle'), 'r') as f:
#         char_to_idx = cPickle.load(f)
#     idx_to_char = { v: k for k, v in char_to_idx.iteritems() }
#
#     n_words = len(char_to_idx)
#
#     sess = tf.InteractiveSession()
#
#     caption_generator = IM2LATEXCaptionGenerator(
#         n_words=char_to_idx, 
#         dim_ctx=dim_ctx,
#         dim_hidden=dim_hidden,
#         n_lstm_steps=max_lstm_steps,
#         batch_size=batch_size,
#         img_shape=img_shape,
#         dropout=1.0)
#
#     images, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=max_lstm_steps)
#     saver = tf.train.Saver()
#     saver.restore(sess, os.path.join(model_path, model_name))
#
#     generated_words = sess.run(generated_words, feed_dict = { images: np.expand_dims(images_data[idx, :, :], axis=0) })
#     translated_words = [ idx_to_char[i] for i in generated_words ]
#
#     print translated_words
#
#
# def sample(model_name="mnist-10"):
#
#     images_data = load_images()
#     idx = np.random.randint(0, high=images_data.shape[0]-1)
#
#     with open(os.path.join(model_path, 'dictionary.pickle'), 'r') as f:
#         char_to_idx = cPickle.load(f)
#     idx_to_char = { v: k for k, v in char_to_idx.iteritems() }
#
#     n_words = len(char_to_idx)
#
#     sess = tf.InteractiveSession()
#
#     caption_generator = IM2LATEXCaptionGenerator(
#         n_words=char_to_idx, 
#         dim_ctx=dim_ctx,
#         dim_hidden=dim_hidden,
#         n_lstm_steps=max_lstm_steps,
#         batch_size=batch_size,
#         img_shape=img_shape,
#         dropout=1.0)
#
#     images, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=max_lstm_steps)
#     saver = tf.train.Saver()
#     saver.restore(sess, os.path.join(model_path, model_name))
#
#     generated_words = sess.run(generated_words, feed_dict = { images: np.expand_dims(images_data[idx, :, :], axis=0) })
#     translated_words = [ idx_to_char[i] for i in generated_words ]
#
#     print translated_words

# def test(model_name="mnist-10", test_limit=1000):
#     mnist_data = input_data.read_data_sets("data/MNIST", one_hot=True)
#     num_test = mnist_data.test.images.shape[0]
#     mnist_test_images = np.reshape(mnist_data.test.images, (num_test, 28, 28))
#     mnist_test_labels = np.reshape(mnist_data.test.labels, (num_test, 10))
#     mnist_test_labels = np.nonzero(mnist_test_labels)[1] # one hot to integer
#     mnist_test_labels = np.reshape(mnist_test_labels, (num_test/CONCAT_LENGTH, CONCAT_LENGTH))
#
#     stacked = np.stack([horizontally_stack(m, 2) for m in np.split(mnist_test_images, num_test/CONCAT_LENGTH, axis=0)])
#
#     sess = tf.InteractiveSession()
#
#     caption_generator = IM2LATEXCaptionGenerator(
#         n_words=10, # 10 possible words
#         dim_ctx=dim_ctx,
#         dim_hidden=dim_hidden,
#         n_lstm_steps=CONCAT_LENGTH,
#         batch_size=batch_size,
#         img_shape=img_shape,
#         dropout=1.0)
#
#     images, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=CONCAT_LENGTH)
#     saver = tf.train.Saver()
#     saver.restore(sess, os.path.join(model_path, model_name))
#
#     num_test = min(test_limit, stacked.shape[0])
#     correct = defaultdict(int)
#     total = defaultdict(int)
#
#     idx_list = np.arange(stacked.shape[0])
#     np.random.shuffle(idx_list)
#     idx_list = idx_list[:num_test] # limit if needed
#
#     for i in tqdm.tqdm(idx_list):
#         generated = sess.run(generated_words, feed_dict = { images: np.expand_dims(stacked[i, :, :], axis=0) })
#         for j in range(len(generated)):
#             total += 1
#             correct += 1 if generated[j][0] == mnist_test_labels[i, j] else 0
#
#     for j in range(CONCAT_LENGTH):
#         print "Precision-{}: {} ({}/{})".format(j, 
#             float(correct[j])/float(total[j]), correct[j], total[j]) 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # sample(sys.argv[1])
        test(sys.argv[1])
    else:
        train()

