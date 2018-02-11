

```python
# 小批量随机梯度下降迭代次数
hm_epochs = 20

# 小批量随机梯度下降每次迭代的数据数量
batch_size = 25
```


```python
# 读取mnist数据集
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)
```


```python
import tensorflow as tf

# x占位符：张量[None, 784]表示第一维可以有任意多个输入，第二位表示长度为784的图片
x = tf.placeholder(tf.float32, [None, 784])

# 权重变量
W = tf.Variable(tf.zeros([784, 10]))

# 偏置变量
b = tf.Variable(tf.zeros([10]))

# 输出层运用softmax算法
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 占位符：表示x的实际标签值
y_ = tf.placeholder("float", [None, 10])
```


```python
# 定义操作：softmax的交叉熵代价函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 定义操作：选择梯度下降算法优化器，学习率为0.01，代价函数为softmax的交叉熵代价函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

# 定义操作：获取预测准确的值，得tf.argmax返回最大的那个数值所在的下标。
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 定义操作：初始化所有变量
init = tf.global_variables_initializer()
```


```python
# 创建一个会话对象，启动图执行图中的所有操作。
with tf.Session as sess:
    # 执行操作：初始化所有变量
    sess.run(init)
    
    for epoch in range(hm_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / float(batch_size))):
            # 获取大小为batch_size的小批量数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            # 执行操作：执行优化和交叉熵
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
            
            # 累积误差
            epoch_loss += c
            
            print "Epoch %d/%d, loss: %s" % (epoch + 1, hm_epochs, epoch_loss)
    
    # 执行操作：找出测试数据集中那些预测正确的标签。
    correct_predictions = sess.run(correct_predictions, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    
    # 确定正确预测项的比例，把布尔值转换成浮点数，然后取平均值。
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
    
    print "Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
```
