try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
import numpy as np
from .BenchmarkModel import BenchmarkModel

'''
class AlexNet(BenchmarkModel):
    # result display data
    DISPLAY_GROUP="FULL_NETWORKS"
    DISPLAY_GROUP_ENUM=1
    # Display Training results

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=112
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"
    DTYPE=tf.float16
    DATA_SHAPE=(1,112,112,1)

    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(self.DTYPE,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS),name="TrainingInput")
        
        self.train_labels = tf.placeholder(self.DTYPE,
                                      shape=(
                                          None,
                                          self.NUM_LABELS),name="TrainingLabels")
        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(self.DTYPE,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS)
                               ,name="input")
        return self.x

    def initVariables(self):
        self.variables = []
        # Layer 1 Conv 
        filter_size = 11
        layer_depth = 96
        self.cw1 = tf.Variable(
            tf.truncated_normal([filter_size,filter_size,1,layer_depth],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer1Weights")
        self.cb1 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[layer_depth]),name="Layer1Bias")
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        # Layer 2 Conv 
        filter_size = 5
        prev_layer_depth = layer_depth
        layer_depth = 256
        self.cw2 = tf.Variable(
            tf.truncated_normal([filter_size,filter_size,prev_layer_depth,layer_depth],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer2Weights")
        self.cb2 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[layer_depth]),name="Layer2Bias")
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        # Layer 3 Conv 
        filter_size = 3
        prev_layer_depth = layer_depth
        layer_depth = 512
        self.cw3 = tf.Variable(
            tf.truncated_normal([filter_size,filter_size,prev_layer_depth,layer_depth],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer3Weights")
        self.cb3 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[layer_depth]),name="Layer3Bias")
        self.variables.append(self.cw3)
        self.variables.append(self.cb3)

        # Layer 4 Conv 
        filter_size = 3
        prev_layer_depth = layer_depth
        layer_depth = 1024
        self.cw4 = tf.Variable(
            tf.truncated_normal([filter_size,filter_size,prev_layer_depth,layer_depth],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer4Weights")
        self.cb4 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[layer_depth]),name="Layer4Bias")
        self.variables.append(self.cw4)
        self.variables.append(self.cb4)

        # Layer 5 Conv 
        filter_size = 3
        prev_layer_depth = layer_depth
        layer_depth = 1024
        self.cw5 = tf.Variable(
            tf.truncated_normal([filter_size,filter_size,prev_layer_depth,layer_depth],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer5Weights")
        self.cb5 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[layer_depth]),name="Layer5Bias")
        self.variables.append(self.cw5)
        self.variables.append(self.cb5)

        # Layer 6 Flatten 
        outSize=3072
        self.cw6 = tf.Variable(
            tf.truncated_normal([1024,outSize],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer6Weights")
        self.cb6 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[outSize]),name="Layer6Bias")
        self.variables.append(self.cw6)
        self.variables.append(self.cb6)

        # Layer 7 
        inSize = outSize
        outSize = 4096
        self.cw7 = tf.Variable(
            tf.truncated_normal([inSize,outSize],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer7Weights")
        self.cb7 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[outSize]),name="Layer7Bias")
        self.variables.append(self.cw7)
        self.variables.append(self.cb7)

      # Layer 8 
        inSize = outSize
        outSize = self.NUM_LABELS
        self.cw8 = tf.Variable(
            tf.truncated_normal([inSize,outSize],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42),name="Layer8Weights")
        self.cb8 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[outSize]),name="Layer8Bias")
        self.variables.append(self.cw8)
        self.variables.append(self.cb8)

        return self.variables

    def conv2d(self,x,W,b,stride=3,padding='SAME'):
        return tf.nn.relu(tf.nn.conv2d(x,W,[1,stride,stride,1],padding,use_cudnn_on_gpu=self.CUDNN)+b)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            x = self.maxpool2d(self.conv2d(x,self.cw1,self.cb1,stride=11))
            x = self.maxpool2d(self.conv2d(x,self.cw2,self.cb2,stride=5))
            x = self.maxpool2d(self.conv2d(x,self.cw3,self.cb3,stride=3))
            x = self.conv2d(x,self.cw4,self.cb4,stride=3)
            x = self.maxpool2d(self.conv2d(x,self.cw5,self.cb5,stride=3))
            x = tf.reshape(x, [-1,1024],name="output")
            x = tf.nn.relu(tf.matmul(x,self.cw6)+self.cb6)
            x = tf.nn.relu(tf.matmul(x,self.cw7)+self.cb7)
            x = tf.matmul(x,self.cw8)+self.cb8
        self.train_y = tf.nn.softmax(x,name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

    def preprocessData(self,data):
        data = super(self.__class__,self).preprocessData(data)
        shape = list(data.shape)
        data = np.resize(data,(shape[0],112,112,shape[3]))
        return data
'''