#-*- coding: utf-8 -*-
import math
import os
import ipdb
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle

import tensorflow.python.platform
from keras.preprocessing import sequence

class Caption_Generator():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_custom_weight(self, shape, name=None, stddev=0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name))

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_4x4(x):
      return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    def _init_conv_net(self):
        # starting image size: 128 x 1024
        # after 5 max pools: 4 x 32 = 128 total pixels
        # with a filter of depth 512, the context shape is 128 x 512

        # copy vgg16 architecture 

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

        with tf.variable_scope("conv5"): 
            self.conv5_W_1 = self.init_custom_weight([3, 3, 512, 512], name="W_1")
            self.conv5_b_1 = self.init_bias(512, name="b_1")
            self.conv5_W_2 = self.init_custom_weight([3, 3, 512, 512], name="W_2")
            self.conv5_b_2 = self.init_bias(512, name="b_2")
            self.conv5_W_3 = self.init_custom_weight([3, 3, 512, 512], name="W_3")
            self.conv5_b_3 = self.init_bias(512, name="b_3")

    def __init__(self, n_words, dim_embed, dim_ctx, dim_hidden, n_lstm_steps, batch_size=200, img_shape=[128,1024], bias_init_vector=None):
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_ctx = dim_ctx
        self.dim_hidden = dim_hidden
        self.img_shape = img_shape 
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size

        self._init_conv_net()

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0), name='Wemb')

        self.init_hidden_W = self.init_weight(dim_ctx, dim_hidden, name='init_hidden_W')
        self.init_hidden_b = self.init_bias(dim_hidden, name='init_hidden_b')

        self.init_memory_W = self.init_weight(dim_ctx, dim_hidden, name='init_memory_W')
        self.init_memory_b = self.init_bias(dim_hidden, name='init_memory_b')

        self.lstm_W = self.init_weight(dim_embed, dim_hidden*4, name='lstm_W')
        self.lstm_U = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U')
        self.lstm_b = self.init_bias(dim_hidden*4, name='lstm_b')

        self.image_encode_W = self.init_weight(dim_ctx, dim_hidden*4, name='image_encode_W')

        self.image_att_W = self.init_weight(dim_ctx, dim_ctx, name='image_att_W')
        self.hidden_att_W = self.init_weight(dim_hidden, dim_ctx, name='hidden_att_W')
        self.pre_att_b = self.init_bias(dim_ctx, name='pre_att_b')

        self.att_W = self.init_weight(dim_ctx, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')

        self.decode_lstm_W = self.init_weight(dim_hidden, dim_embed, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(dim_embed, name='decode_lstm_b')

        self.decode_word_W = self.init_weight(dim_embed, n_words, name='decode_word_W')

        if bias_init_vector is not None:
            self.decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
        else:
            self.decode_word_b = self.init_bias(n_words, name='decode_word_b')


    def get_initial_lstm(self, mean_context):
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    # TODO(yoavz): do we need to add scopes?
    def max_pool(self, layer):
        return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def conv_layer(self, layer, filt, bias):
        conv = tf.nn.conv2d(layer, filt, [1, 1, 1, 1], padding='SAME')
        biases = tf.nn.bias_add(conv, bias)
        relu = tf.nn.relu(biases)
        return relu

    def build_conv_net(self, images):

        # reshape images into 4 rank for convolution operation: [ batch, h, w, 1]
        images_shaped = tf.expand_dims(images, -1)

        # 128 x 1024
        conv1_1 = self.conv_layer(images_shaped, self.conv1_W_1, self.conv1_b_1)
        conv1_2 = self.conv_layer(conv1_1, self.conv1_W_2, self.conv1_b_2)
        pool1 = self.max_pool(conv1_2)

        # 64 x 512
        assert pool1.get_shape().as_list()[1:] == [64, 512, 64]

        conv2_1 = self.conv_layer(pool1, self.conv2_W_1, self.conv2_b_1)
        conv2_2 = self.conv_layer(conv2_1, self.conv2_W_2, self.conv2_b_2)
        pool2 = self.max_pool(conv2_2)

        # 32 x 256
        assert pool2.get_shape().as_list()[1:] == [32, 256, 128]

        conv3_1 = self.conv_layer(pool2, self.conv3_W_1, self.conv3_b_1)
        conv3_2 = self.conv_layer(conv3_1, self.conv3_W_2, self.conv3_b_2)
        conv3_3 = self.conv_layer(conv3_2, self.conv3_W_3, self.conv3_b_3)
        pool3 = self.max_pool(conv3_3)

        # 16 x 128
        assert pool3.get_shape().as_list()[1:] == [16, 128, 256]

        conv4_1 = self.conv_layer(pool3, self.conv4_W_1, self.conv4_b_1)
        conv4_2 = self.conv_layer(conv4_1, self.conv4_W_2, self.conv4_b_2)
        conv4_3 = self.conv_layer(conv4_2, self.conv4_W_3, self.conv4_b_3)
        pool4 = self.max_pool(conv4_3)

        # 8 x 64
        assert pool4.get_shape().as_list()[1:] == [8, 64, 512]

        conv5_1 = self.conv_layer(pool4, self.conv5_W_1, self.conv5_b_1)
        conv5_2 = self.conv_layer(conv5_1, self.conv5_W_2, self.conv5_b_2)
        conv5_3 = self.conv_layer(conv5_2, self.conv5_W_3, self.conv5_b_3)
        pool5 = self.max_pool(conv5_3)

        # 4 x 32 (x 512 filters)
        assert pool5.get_shape().as_list()[1:] == [4, 32, 512]

        return tf.reshape(pool5, [-1, 128, 512])

    def build_model(self):
        images = tf.placeholder("float32", [self.batch_size, self.img_shape[0], self.img_shape[1]], name="images")
        sentence = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps], name="sentences")
        mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps], name="mask")

        # (yoavz): add conv layers to get the image contexts
        context = self.build_conv_net(images)
        ctx_shape = (context.get_shape().as_list()[1], context.get_shape().as_list()[2])

        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

        # TensorFlow가 dot(3D tensor, matrix) 계산을 못함;;; ㅅㅂ 삽질 ㄱㄱ
        context_flat = tf.reshape(context, [-1, self.dim_ctx])
        context_encode = tf.matmul(context_flat, self.image_att_W) # (batch_size, 196, 512)
        context_encode = tf.reshape(context_encode, [-1, ctx_shape[0], ctx_shape[1]])

        loss = 0.0

        for ind in range(self.n_lstm_steps):

            if ind == 0:
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
            else:
                tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"):
                    word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,ind-1])

            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b # (batch_size, hidden*4)

            labels = tf.expand_dims(sentence[:,ind], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense( concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

            context_encode = context_encode + \
                 tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + \
                 self.pre_att_b

            context_encode = tf.nn.tanh(context_encode)

            # 여기도 context_encode: 3D -> flat required
            context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx]) # (batch_size*196, 512)
            alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b # (batch_size*196, 1)
            alpha = tf.reshape(alpha, [-1, ctx_shape[0]])
            alpha = tf.nn.softmax( alpha )

            weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)
            i, f, o, new_c = tf.split(1, 4, lstm_preactive)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(new_c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)
            # TODO: add dropout back in ?
            logits = tf.nn.dropout(logits, 1.0)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            cross_entropy = cross_entropy * mask[:,ind]

            current_loss = tf.reduce_sum(cross_entropy)
            loss = loss + current_loss

        loss = loss / tf.reduce_sum(mask)
        return loss, images, sentence, mask

    def build_generator(self, maxlen):
        images = tf.placeholder("float32", [1, self.img_shape[0], self.img_shape[1]])

        #(yoavz): feed image through convnet
        context = self.build_conv_net(images)
        ctx_shape = (context.get_shape().as_list()[1], context.get_shape().as_list()[2])

        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

        context_encode = tf.matmul(tf.squeeze(context), self.image_att_W)
        generated_words = []
        logit_list = []
        alpha_list = []
        word_emb = tf.zeros([1, self.dim_embed])
        for ind in range(maxlen):
            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b
            context_encode = context_encode + tf.matmul(h, self.hidden_att_W) + self.pre_att_b
            context_encode = tf.nn.tanh(context_encode)

            alpha = tf.matmul(context_encode, self.att_W) + self.att_b
            alpha = tf.reshape(alpha, [-1, ctx_shape[0]] )
            alpha = tf.nn.softmax(alpha)

            alpha = tf.reshape(alpha, (ctx_shape[0], -1))
            alpha_list.append(alpha)

            weighted_context = tf.reduce_sum(tf.squeeze(context) * alpha, 0)
            weighted_context = tf.expand_dims(weighted_context, 0)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)

            i, f, o, new_c = tf.split(1, 4, lstm_preactive)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f*c + i*new_c
            h = o*tf.nn.tanh(new_c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b

            max_prob_word = tf.argmax(logit_words, 1)

            with tf.device("/cpu:0"):
                word_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

            generated_words.append(max_prob_word)
            logit_list.append(logit_words)

        return images, generated_words, logit_list, alpha_list


def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


###### 학습 관련 Parameters ######
n_epochs=1000
batch_size=80
dim_embed=256
dim_ctx=512
dim_hidden=256
# ctx_shape=[196,512]
img_shape=[128,1024]
pretrained_model_path = './model/model-8'
#############################
###### 잡다한 Parameters #####
annotation_path = './data/annotations.pickle'
feat_path = './data/feats.npy'
model_path = './model/'
#############################

def train(pretrained_model_path=pretrained_model_path): # 전에 학습하던게 있으면 초기값 설정.
    # TODO(yoavz): rewrite this function
    annotation_data = pd.read_pickle(annotation_path)
    captions = annotation_data['caption'].values
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

    learning_rate=0.001
    n_words = len(wordtoix)
    feats = np.load(feat_path)
    maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )

    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen+1, # w1~wN까지 예측한 뒤 마지막에 '.'예측해야하니까 +1
            batch_size=batch_size,
            img_shape=img_shape,
            bias_init_vector=bias_init_vector)

    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()
    if pretrained_model_path is not None:
        print "Starting with pretrained model"
        saver.restore(sess, pretrained_model_path)

    index = list(annotation_data.index)
    np.random.shuffle(index)
    annotation_data = annotation_data.ix[index]

    captions = annotation_data['caption'].values
    image_id = annotation_data['image_id'].values

    for epoch in range(n_epochs):
        for start, end in zip( \
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

            current_feats = feats[ image_id[start:end] ]
            current_feats = current_feats.reshape(-1, img_shape[1], img_shape[0]).swapaxes(1,2)

            current_captions = captions[start:end]
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions) # '.'은 제거

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                context:current_feats,
                sentence:current_caption_matrix,
                mask:current_mask_matrix})

            print "Current Cost: ", loss_value
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

def test(test_feat='./guitar_player.npy', model_path='./model/model-6', maxlen=20):
    annotation_data = pd.read_pickle(annotation_path)
    captions = annotation_data['caption'].values
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
    n_words = len(wordtoix)
    feat = np.load(test_feat).reshape(-1, img_shape[1], img_shape[0]).swapaxes(1,2)

    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen,
            batch_size=batch_size,
            img_shape=img_shape)

    context, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=maxlen)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index = sess.run(generated_words, feed_dict={context:feat})
    alpha_list_val = sess.run(alpha_list, feed_dict={context:feat})
    generated_words = [ixtoword[x[0]] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1

    generated_words = generated_words[:punctuation]
    alpha_list_val = alpha_list_val[:punctuation]
    return generated_words, alpha_list_val

#    generated_sentence = ' '.join(generated_words)
#    ipdb.set_trace()

if __name__ == "__main__":

    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(
            n_words=1000,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=1000,
            batch_size=batch_size,
            img_shape=img_shape)

    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)

