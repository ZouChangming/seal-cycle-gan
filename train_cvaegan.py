#encoding=utf-8

import os
import tensorflow as tf
from models import Generative, Discriminative, Encoder, Classifier, Generative2, Discriminative2
import tools

learning_rate = 0.00001
batch_size = 8
model_path = './data/model/GAN'
epoch = 10000
iter = 10

def train():
    seal = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    noseal = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
    seal_label = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    noseal_label = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    global_step = tf.Variable(0)

    # z = Encoder(seal)
    fake_noseal = Generative2(seal, 'seal2noseal')
    fake_seal = Generative2(noseal, 'noseal2seal')
    fake_noseal2seal = Generative2(fake_noseal, 'noseal2seal', reuse=True)
    fake_seal2noseal = Generative2(fake_seal, 'seal2noseal', reuse=True)

    D_logits_fake_noseal = Discriminative(fake_noseal)
    D_logits_fake_seal = Discriminative(fake_seal, reuse=True)
    D_logits_real_seal = Discriminative(seal, reuse=True)
    D_logits_real_noseal = Discriminative(noseal, reuse=True)

    _, C_logits_fake_noseal = Classifier(fake_noseal)
    _, C_logits_fake_seal = Classifier(fake_seal, reuse=True)
    _, C_logits_real_noseal = Classifier(noseal, reuse=True)
    _, C_logits_real_seal = Classifier(seal, reuse=True)

    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logits_real_seal), logits=D_logits_real_seal) + \
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logits_real_noseal), logits=D_logits_real_noseal) + \
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logits_fake_noseal), logits=D_logits_fake_noseal)  +\
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logits_fake_seal), logits=D_logits_fake_seal))

    loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=seal_label, logits=C_logits_real_seal) + \
                tf.nn.softmax_cross_entropy_with_logits(labels=noseal_label, logits=C_logits_real_noseal))

    loss_GD_seal2noseal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logits_fake_noseal),
                                                                                 logits=D_logits_fake_noseal))
    loss_GD_noseal2seal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logits_fake_seal),
                                                                                 logits=D_logits_fake_seal))

    loss_GC_seal2noseal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=noseal_label, logits=C_logits_fake_noseal))

    loss_GC_noseal2seal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=seal_label, logits=C_logits_fake_seal))

    loss_G = tools.get_loss_G(seal, fake_noseal2seal) + tools.get_loss_G(noseal, fake_seal2noseal)

    # loss_KL = tools.get_loss_KL(z)

    all_var = tf.trainable_variables()

    var_C = [var for var in all_var if var.name.startswith('Classifier')]

    # var_E = [var for var in all_var if var.name.startswith('Encoder')]

    var_G = [var for var in all_var if var.name.startswith('Generative')]

    var_G_seal2noseal = [var for var in all_var if var.name.startswith('Generative_seal2noseal')]

    var_G_noseal2seal = [var for var in all_var if var.name.startswith('Generative_noseal2seal')]

    var_D = [var for var in all_var if var.name.startswith('Discriminative')]

    lr = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_rate=0.5,
                                    decay_steps=5000, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)

    opt_C = optimizer.minimize(loss_C, var_list=var_C, global_step=global_step)

    # opt_E = optimizer.minimize(3*loss_KL + loss_G, var_list=var_E)
    opt_G_seal2noseal = optimizer.minimize(0.1*loss_GC_seal2noseal + loss_GD_seal2noseal, var_list=var_G_seal2noseal)

    opt_G_noseal2seal = optimizer.minimize(0.1*loss_GC_noseal2seal + loss_GD_noseal2seal, var_list=var_G_noseal2seal)

    opt_G = optimizer.minimize(15*loss_G, var_list=var_G)

    opt_D = optimizer.minimize(loss_D, var_list=var_D)

    accuracy = 0.5*(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(C_logits_real_seal), 1),
                                                    tf.argmax(tf.nn.softmax(seal_label), 1)), tf.float32)) + \
                    tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(C_logits_real_noseal), 1),
                                                    tf.argmax(tf.nn.softmax(noseal_label), 1)), tf.float32)))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            pass

        for i in range(1, epoch+1):
            for j in range(iter):
                seal_img, seal_labels = tools.get_seal_set(batch_size/2, i*iter+j)
                noseal_img, noseal_labels = tools.get_noseal_set(batch_size/2, i*iter+j)
                # sess.run(opt_E, feed_dict={seal:seal_img})
                sess.run(opt_G, feed_dict={seal:seal_img, seal_label:seal_labels, noseal:noseal_img,
                                           noseal_label:noseal_labels})
                sess.run(opt_D, feed_dict={seal:seal_img, seal_label:seal_labels, noseal:noseal_img,
                                           noseal_label:noseal_labels})
                sess.run(opt_C, feed_dict={seal:seal_img, seal_label:seal_labels, noseal:noseal_img,
                                           noseal_label:noseal_labels})
                sess.run(opt_G_seal2noseal, feed_dict={seal: seal_img, seal_label: seal_labels, noseal: noseal_img,
                                           noseal_label: noseal_labels})
                sess.run(opt_G_noseal2seal, feed_dict={seal: seal_img, seal_label: seal_labels, noseal: noseal_img,
                                           noseal_label: noseal_labels})
            # tmp = sess.run(logits_fake, feed_dict={seal:seal_img, seal_label:seal_labels,
            #                                         noseal:noseal_img,noseal_label:noseal_labels})
            # print tmp
            l_c, l_g, l_c1, l_c2, l_d1, l_d2, l_d, a, l = sess.run([loss_C, loss_G, loss_GC_seal2noseal, loss_GC_noseal2seal,
                                                        loss_GD_seal2noseal, loss_GD_noseal2seal, loss_D, accuracy, lr],
                                                          feed_dict={seal:seal_img, seal_label:seal_labels,
                                                                     noseal:noseal_img,noseal_label:noseal_labels})
            print 'Epoch ' + str(i) + ':L_C=' + str(l_c) + ';L_G=' + str(l_g) + ';L_C_s2n=' + str(l_c1) +  ';L_C_n2s=' +\
                  str(l_c2) + ';L_D_s2n=' + str(l_d1) + ';L_D_n2s=' + str(l_d2) + ';L_D=' + str(l_d) + ';acc=' + str(a) +\
                  ';lr=' + str(l)
            if i % 10 == 0:
                saver.save(sess, os.path.join(model_path, 'model.ckpt'))
                seal_img = tools.get_test()
                fake_img = sess.run(fake_noseal, feed_dict={seal:seal_img})
                tools.save(fake_img[0, :, : ,:], i)

if __name__ == '__main__':
    train()