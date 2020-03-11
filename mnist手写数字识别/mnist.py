#coding=utf-8

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

mnist=input_data.read_data_sets("./materials",one_hot=True)

#显示数据
plt.figure(figsize=(8,8))
for idx in range(16):
    plt.subplot(4,4,idx+1)
    plt.axis('off')
    plt.title('[{}]'.format(np.argmax(mnist.train.labels[idx])))
    plt.imshow(mnist.train.images[idx].reshape((28,28)))

x = tf.placeholder( "float", [None, 784], name='x' )
y = tf.placeholder( "float", [None, 10], name='y' )

# 因为输入的图片是展开后的一维向量，所以需要把一维向量还原成二维图片
x_image = tf.reshape( x, [-1, 28, 28, 1] )

# 定义第一个卷积层
with tf.name_scope( 'conv1' ):
    C1 = tf.contrib.slim.conv2d( x_image, 6, [5, 5], padding='VALID', activation_fn=tf.nn.relu )

# 定义最大值池化
with tf.name_scope( 'pool1' ):
    M2 = tf.contrib.slim.max_pool2d( C1, [2, 2], stride=[2, 2], padding='VALID' )

# 定义第二个卷积层
with tf.name_scope( 'conv2' ):
    C3 = tf.contrib.slim.conv2d( M2, 16, [5, 5], padding='VALID', activation_fn=tf.nn.relu )

# 定义最大值池化
with tf.name_scope( 'pool2' ):
    M4 = tf.contrib.slim.max_pool2d( C3, [2, 3], stride=[2, 2], padding='VALID' )

# 定义两个全连接
with tf.name_scope( 'fc1' ):
    M4_flat = tf.contrib.slim.flatten( M4 )
    FC5 = tf.contrib.slim.fully_connected( M4_flat, 120, activation_fn=tf.nn.relu )

with tf.name_scope( 'fc2' ):
    FC6 = tf.contrib.slim.fully_connected( FC5, 84, activation_fn=tf.nn.relu )

# 为防止过拟合，添加一个0.6的dropout，以40%的概率关闭全连接层中的神经元
# 需要注意的是，dropout仅在训练的时候使用，验证的时候，需要关闭dropout，所以验证时候的keep_prob是1.0。
# dropout的输出最终送入一个隐层为10的全连接层，这个全连接层即为最后的分类器
with tf.name_scope( 'dropout' ):
    keep_prob = tf.placeholder( name='keep_prob', dtype=tf.float32 )
    FC6_drop = tf.nn.dropout( FC6, keep_prob )

with tf.name_scope( 'fc3' ):
    output = tf.contrib.slim.fully_connected( FC6_drop, 10, activation_fn=None )


cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))

l2_loss=tf.add_n([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(w.name)
    tf.summary.histogram(w.name,w)  #对权重进行监控（histogram适用于矩阵）

total_loss=cross_entropy_loss+7e-5*l2_loss
tf.summary.scalar('cross_entropy_loss',cross_entropy_loss)
tf.summary.scalar('l2_loss',l2_loss)
tf.summary.scalar('total_loss',total_loss)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(total_loss)


pred=tf.nn.softmax(output)
correct=tf.equal(tf.argmax(y,1),tf.arg_max(pred,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

batch_size = 100
trainig_step = 1100

#保存训练参数
saver = tf.train.Saver()

merged = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter( "logs/", sess.graph )
    sess.run( tf.global_variables_initializer() )

    # 定义验证集与测试集
    validate_data = {x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.0}
    test_data = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}

    for i in range( trainig_step ):
        xs, ys = mnist.train.next_batch( batch_size )
        _, loss, rs = sess.run( [optimizer, cross_entropy_loss, merged], feed_dict={x: xs, y: ys, keep_prob: 0.6} )

        writer.add_summary( rs, i )

        # 每隔100次训练，打印一次损失值与验证准确率
        if i > 0 and i % 100 == 0:
            validate_accuray = sess.run( accuracy, feed_dict=validate_data )
            print( "After %d training steps, the loss is %g,the validation is %g" % (i, loss, validate_accuray) )
            #saver.save( sess, './model.ckpt', global_step=i )

    print( 'Training is finished.' )
    acc = sess.run( accuracy, feed_dict=test_data )
    print( "The test accuracy is : ", acc )












