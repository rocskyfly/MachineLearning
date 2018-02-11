

```python
import tensorflow as tf

def neural_network_mode(x_train, sizes=list((784, 400, 10))):
    """定义神经网络模型"""
    num_layers = len(sizes)
    activation = x_train
    for layer in range(num_layers - 1):
        layer = {'weights': tf.Variable(tf.random_normal([sizes[layer], sizes[layer + 1]])),
                 'biases': tf.Variable(tf.random_normal([sizes[layer + 1]]))}

        # input_data * weights + biases; define activation function
        activation = tf.add(tf.matmul(activation, layer['weights']), layer['biases'])
        activation = tf.nn.sigmoid(activation)

    return activation
```


```python
def train_neural_network(x_train, y_train, sizes, batch_size=25, hm_epochs=20):
    output = neural_network_mode(x_train, sizes)

    # 定义操作：softmax的交叉熵代价函数。
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_train))

    # 定义操作：选择AdamOptimizer（A Method for Stochastic Optimization）。
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # 定义操作：初始化所有变量。
    init = tf.global_variables_initializer()

    # 创建一个会话对象，启动图执行图中的所有操作。
    with tf.Session() as sess:
        # 执行操作：初始化所有变量
        sess.run(init)

        # train neural network
        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / float(batch_size))):
                # 获取大小为batch_size的小批量数据
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # 执行操作：执行优化和交叉熵
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                # 累积误差
                epoch_loss += c

            print "Epoch %d/%d, loss: %f" % (epoch + 1, hm_epochs, epoch_loss)

        # 找出测试数据集中那些预测正确的标签。
        correct_predictions = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

        # 确定正确预测项的比例，把布尔值转换成浮点数，然后取平均值。
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

        print "Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
```


```python
if __name__ == '__main__':
    # 读取mnist数据集
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    mnist = read_data_sets("MNIST_data/", one_hot=True)

    # 定义神经网络结构
    layers = [784, 600, 10]

    # height x width
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')

    train_neural_network(x, y, layers)
```
