try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
    
from .BenchmarkModel import BenchmarkModel

class CNN5_FLOAT32(BenchmarkModel):
    """
    A net with 5 convolutions and maxpools
    """
    # result display data
    DISPLAY_GROUP="SMALL_NETS"
    DISPLAY_GROUP_ENUM=1

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"
    if not IS_DUMMY:
        DTYPE=tf.float32
       
    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(self.DTYPE,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS))
        
        self.train_labels = tf.placeholder(self.DTYPE,
                                      shape=(
                                          None,
                                          self.NUM_LABELS))
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
        self.cw1 = tf.Variable(
            tf.truncated_normal([3,3,1,32],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42))
        self.cb1 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[32]))
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        self.cw2 = tf.Variable(
            tf.truncated_normal([3,3,32,64],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42))
        self.cb2 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[64]))
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        self.cw3 = tf.Variable(
            tf.truncated_normal([3,3,64,128],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42))
        self.cb3 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[128]))
        self.variables.append(self.cw3)
        self.variables.append(self.cb3)
        self.cw4 = tf.Variable(
            tf.truncated_normal([3,3,128,64],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42))
        self.cb4 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[64]))
        self.variables.append(self.cw4)
        self.variables.append(self.cb4)
        self.cw5 = tf.Variable(
            tf.truncated_normal([7,7,64,self.NUM_LABELS],
                stddev=0.1,
                dtype=self.DTYPE,
                seed=42))
        self.cb5 = tf.Variable(tf.constant(0.1,dtype=self.DTYPE,shape=[self.NUM_LABELS]))
        self.variables.append(self.cw5)
        self.variables.append(self.cb5)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding,use_cudnn_on_gpu=self.CUDNN)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = tf.nn.relu(self.conv2d(x,self.cw1,padding='SAME') + self.cb1)
            self.convOut = self.maxpool2d(self.convOut)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw2,padding='SAME') + self.cb2)
            self.convOut = self.maxpool2d(self.convOut)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw3,padding='SAME') + self.cb3)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw4,padding='SAME') + self.cb4)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw5,padding='VALID') + self.cb5)
        #Flatten
        self.train_y = tf.reshape(self.convOut, [-1,self.NUM_LABELS],name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

class CNN5_FLOAT16(CNN5_FLOAT32):
    """
    Same net as the one above, but with half precision
    """
    if not IS_DUMMY: 
        DTYPE=tf.float16