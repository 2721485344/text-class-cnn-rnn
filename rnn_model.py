# coding: utf-8

import tensorflow as tf


class RNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 字的特征是64
    seq_length = 600  # 序列长度    句子长度600
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 字数

    hidden_dim = 128  # 全连接层神经元
    num_layers=2 #隐藏层个数
    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率0.001

    batch_size = 64  # 每批训练大小
    num_epochs = 10000 # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    rnn='gru'

class TextRNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')#句子长度(句子数,600)
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')#标签类别(1,10)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')#设置的dropout

        self.rnn()

    def rnn(self):
        """RNN模型"""  
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)
         
        def dropout():
            if(self.config.rnn=="lstm"):
                cell=lstm_cell()
            else:
                cell=gru_cell()        
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
        # 字向量映射
        with tf.device('/cpu:0'):#5000行64列代表一个字
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])#(5000,64)5000个字
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x) #选取一个张量里面索引对应的元素 shape=(句子数, 600, 64)

        with tf.name_scope("rnn"):
            #多层rnn网络
            cells=[dropout() for _ in range(self.config.num_layers)]
            rnn_cell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            _outputs,_= tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs,dtype=tf.float32)
            last=_outputs[:,-1,:]
       
            ## CNN layer  embedding_inputs 是三维(,h,w)
            #conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            ##1*5的个卷积核，256个核 一维计算 输入的(?, 600, 64)      输出 shape=(?, 596, 256)  (600-5)/1+1=596  
            ## global max pooling layer reduce_max计算张量的各个维度上的元素的最大值  64个句子，每个句子是600 每个字是256维(每个维度是1*5卷积)
            #gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')#shape=(?, 256) 按1维去取最大[[1,2],[3,4]]指定按行列，不指定按均值

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活gmp输入的数据，hidden_dim输出的维度大小
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')#shape=(64, 128)
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)#shape=(64, 128)
            fc = tf.nn.relu(fc)#shape=(64, 128)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')#shape=(?, 10)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别 shape=(?,)按列取

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)# shape=(?,)
            self.loss = tf.reduce_mean(cross_entropy)#shape=() 
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)#shape=(?,)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#shape=()
