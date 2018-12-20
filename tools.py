#encoding=utf-8

import os
import cv2
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

def get_classifier_pic(batch_size, seed):
    random.seed(seed)
    image = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    label = np.zeros(shape=(batch_size, 2), dtype=np.float32)
    for i in range(batch_size):
        flag = random.choice([0, 1])
        if flag > 0:
            path = './data/source/seal'
            img_list = os.listdir(path)
            id = random.randint(0, len(img_list)-1)
            img = cv2.imread(os.path.join(path, img_list[id]))
            img = cv2.resize(img, (256, 256))
            image[i, :, :, :] = img / 255.0 * 2 - 1.0
            label[i, 0] = 1
        else:
            path = './data/source/noseal'
            img_list = os.listdir(path)
            id = random.randint(0, len(img_list)-1)
            img = cv2.imread(os.path.join(path, img_list[id]))
            img = cv2.resize(img, (256, 256))
            image[i, :, :, :] = img / 255.0 * 2 - 1.0
            label[i, 1] = 1
    return image, label

def get_loss_GD(logits_seal, logits_noseal, logits_fake):
    vector_real = 0.5 * (tf.reduce_mean(logits_noseal, 0) + tf.reduce_mean(logits_seal, 0))
    vector_fake = tf.reduce_mean(logits_fake, 0)
    return 0.5*tf.reduce_sum(tf.square(vector_fake - vector_real))

def get_loss_GC(logits_real_noseal, logits_fake_noseal):
    vector_real = tf.reduce_mean(logits_real_noseal, 0)
    vector_fake = tf.reduce_mean(logits_fake_noseal, 0)
    return 0.5*tf.reduce_sum(tf.square(vector_real - vector_fake))

def get_loss_G(seal, fake):
    return tf.reduce_mean(tf.abs(seal - fake))

def get_LS_loss(feature1, feature2):
    return tf.reduce_mean(tf.square(feature1-feature2))

def get_loss_KL(z):
    avg = tf.reduce_mean(z, 1)
    cov = get_covariance(z)
    return 0.5*(tf.reduce_sum(tf.multiply(cov, cov) + tf.exp(avg) - avg - 1))

def get_covariance(vector):
    mean = tf.reduce_mean(vector, 1)
    mean = tf.reshape(mean, [-1, 1])
    vector = tf.subtract(vector, mean)
    return tf.reduce_sum(tf.multiply(vector, vector), 1) / 255.0

def get_seal_set(batch_size, seed):
    random.seed(seed)
    image = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    label = np.zeros(shape=(batch_size, 2), dtype=np.float32)
    path = './data/source/seal'
    img_list = os.listdir(path)
    for i in range(batch_size):
        id = random.randint(0, len(img_list)-1)
        img = cv2.imread(os.path.join(path, img_list[id]))
        img = cv2.resize(img, (256, 256))
        image[i, :, :, :] = img / 255.0 * 2 - 1.0
        label[i, 0] = 1
    return image, label

def get_noseal_set(batch_size, seed):
    random.seed(seed)
    image = np.zeros(shape=(batch_size, 256, 256, 3), dtype=np.float32)
    label = np.zeros(shape=(batch_size, 2), dtype=np.float32)
    path = './data/source/noseal'
    img_list = os.listdir(path)
    for i in range(batch_size):
        id = random.randint(0, len(img_list) - 1)
        img = cv2.imread(os.path.join(path, img_list[id]))
        img = cv2.resize(img, (256, 256))
        image[i, :, :, :] = img / 255.0 * 2 - 1.0
        label[i, 1] = 1
    return image, label

def get_test():
    image = np.zeros(shape=(1, 256, 256, 3), dtype=np.float32)
    path = './data/source/test'
    img_list = os.listdir(path)
    id = random.randint(0, len(img_list)-1)
    img = cv2.imread(os.path.join(path, img_list[id]))
    h, w, _ = np.shape(img)
    img = cv2.resize(img, (256, 256))
    image[0, :, :, :] = img / 255.0 * 2 - 1.0
    return image, h, w

def save(ori_img, image, epoch, h, w):
    img = np.zeros(shape=(h, 2*w, 3), dtype=np.float32)
    image = np.reshape(image, [256, 256, 3])
    image = (image + 1.0) / 2.0 * 255.0
    image = cv2.resize(image, (w, h))
    ori_img = np.reshape(ori_img, [256, 256, 3])
    ori_img = (ori_img + 1.0) / 2.0 * 255.0
    ori_img = cv2.resize(ori_img, (w, h))
    img[:, 0:w, :] = ori_img
    img[:, w:2*w, :] = image
    path = './data/sample/' + str(epoch) + '.jpg'
    cv2.imwrite(path, img)

def check_source():
    path = './data/source/noseal'
    img_list = os.listdir(path)
    for name in img_list:
        print os.path.join(path, name)
        img = cv2.imread(os.path.join(path, name))
        img = cv2.resize(img, (256, 256))
    path = './data/source/seal'
    img_list = os.listdir(path)
    for name in img_list:
        print os.path.join(path, name)
        img = cv2.imread(os.path.join(path, name))
        img = cv2.resize(img, (256, 256))

if __name__ == '__main__':
    check_source()