#encoding=utf-8

import tools
import tensorflow as tf
from models import Generative2
import os
import cv2
import numpy as np

model_path = './data/model/GAN'
dev_path = './data/source/dev'

def verification():
    seal = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)

    fake_noseal = Generative2(seal, 'seal2noseal')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass

        img_list = os.listdir(dev_path)
        for item in img_list:
            img = cv2.imread(os.path.join(dev_path, item))
            h, w, _ = np.shape(img)
            image = np.zeros(shape=(h, w*2, 3), dtype=np.float32)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0 * 2 - 1.0
            img = np.reshape(img, [1, 224, 224, 3])
            fake_img = sess.run(fake_noseal, feed_dict={seal: img})
            fake_img = np.reshape(fake_img, [224, 224, 3])
            fake_img = (fake_img + 1.0) / 2.0 * 255.0
            img = (img + 1.0) / 2.0 * 255.0
            img = np.reshape(img, [224, 224, 3])
            img = cv2.resize(img, (w, h))
            fake_img = cv2.resize(fake_img, (w, h))
            image[:, 0:w, :] = img
            image[:, w:w*2, :] = fake_img
            path = './data/sample_dev/' + item.split('.')[0] + 'B.jpg'
            cv2.imwrite(path, image)

if __name__ == '__main__':
    verification()