try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
    
from .BenchmarkModel import BenchmarkModel

class CNN3Stepped(BenchmarkModel):
    """
    A fully convolutional network (nothing but convolutions used)
    Convolutions used: 
    - 3x3 Stepsize 2
    - 3x3 Stepsize 2
    - 7x7 Stepsize 1
    """
    # result display data
    DISPLAY_GROUP="CNN_EVAL"
    DISPLAY_GROUP_ENUM=1

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"
       
    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS))
        
        self.train_labels = tf.placeholder(tf.float32,
                                      shape=(
                                          None,
                                          self.NUM_LABELS))
        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS)
                               ,name="input")
        return self.x

    def initVariables(self):
        self.cw1 = tf.Variable(
            tf.truncated_normal([3,3,1,16],
                stddev=0.1,
                seed=42))
        self.cb1 = tf.Variable(tf.constant(0.1,shape=[16]))
        self.cw2 = tf.Variable(
            tf.truncated_normal([3,3,16,32],
                stddev=0.1,
                seed=42))
        self.cb2 = tf.Variable(tf.constant(0.1,shape=[32]))
        self.cw3 = tf.Variable(
            tf.truncated_normal([7,7,32,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.cb3 = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables = []
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        self.variables.append(self.cw3)
        self.variables.append(self.cb3)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding,use_cudnn_on_gpu=self.CUDNN)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = tf.nn.relu(self.conv2d(x,self.cw1,padding='SAME',strides=[1,2,2,1]) + self.cb1)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw2,padding='SAME',strides=[1,2,2,1]) + self.cb2)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw3,padding='VALID') + self.cb3)
        #Flatten
        self.train_y = tf.reshape(self.convOut, [-1,self.NUM_LABELS],name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

class CNN3Maxpool(BenchmarkModel):
    """
    A fully convolutional network (nothing but convolutions used)
    Convolutions used: 
    - 3x3 Stepsize 1
    - Maxpool Stepsize 2
    - 3x3 Stepsize 1
    - Maxpool Stepsize 2
    - 7x7 Stepsize 1
    """
    # result display data
    DISPLAY_GROUP="CNN_EVAL"
    DISPLAY_GROUP_ENUM=2

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"

    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS))
        
        self.train_labels = tf.placeholder(tf.float32,
                                      shape=(
                                          None,
                                          self.NUM_LABELS))
        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS),
                               name="input")
        return self.x

    def initVariables(self):
        self.cw1 = tf.Variable(
            tf.truncated_normal([3,3,1,16],
                stddev=0.1,
                seed=42))
        self.cb1 = tf.Variable(tf.constant(0.1,shape=[16]))
        self.cw2 = tf.Variable(
            tf.truncated_normal([3,3,16,32],
                stddev=0.1,
                seed=42))
        self.cb2 = tf.Variable(tf.constant(0.1,shape=[32]))
        self.cw3 = tf.Variable(
            tf.truncated_normal([7,7,32,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.cb3 = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables = []
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        self.variables.append(self.cw3)
        self.variables.append(self.cb3)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding,use_cudnn_on_gpu=self.CUDNN)
    
    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = tf.nn.relu(self.conv2d(x,self.cw1,padding='SAME',strides=[1,1,1,1]) + self.cb1)
            self.convOut = self.maxpool2d(self.convOut)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw2,padding='SAME',strides=[1,1,1,1]) + self.cb2)
            self.convOut = self.maxpool2d(self.convOut)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw3,padding='VALID') + self.cb3)
        #Flatten
        self.train_y = tf.reshape(self.convOut, [-1,self.NUM_LABELS],name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

class CNN2FC1(BenchmarkModel):
    """
    A convolutional network with fully connected layer at the end
    Convolutions used: 
    - 3x3 Stepsize 2
    - 3x3 Stepsize 2
    - Flatten and fully connect from 7x7x32 to 10x1
    """
    # result display data
    DISPLAY_GROUP="CNN_EVAL"
    DISPLAY_GROUP_ENUM=3

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"

    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS))
        
        self.train_labels = tf.placeholder(tf.float32,
                                      shape=(
                                          None,
                                          self.NUM_LABELS))
        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS),
                               name="input")
        return self.x

    def initVariables(self):
        self.cw1 = tf.Variable(
            tf.truncated_normal([3,3,1,16],
                stddev=0.1,
                seed=42))
        self.cb1 = tf.Variable(tf.constant(0.1,shape=[16]))
        self.cw2 = tf.Variable(
            tf.truncated_normal([3,3,16,32],
                stddev=0.1,
                seed=42))
        self.cb2 = tf.Variable(tf.constant(0.1,shape=[32]))
        self.aw1 = tf.Variable(
            tf.truncated_normal([7*7*32,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.ab1 = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables = []
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        self.variables.append(self.aw1)
        self.variables.append(self.ab1)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding,use_cudnn_on_gpu=self.CUDNN)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = tf.nn.relu(self.conv2d(x,self.cw1,padding='SAME',strides=[1,2,2,1]) + self.cb1)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw2,padding='SAME',strides=[1,2,2,1]) + self.cb2)
            #Flatten
            data_shape = self.convOut.get_shape().as_list()
            reshaped = tf.reshape(self.convOut, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
            end = tf.matmul(reshaped,self.aw1) + self.ab1
        self.train_y = tf.nn.relu(end,name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

class CNN2MaxpoolFC1(BenchmarkModel):
    """
    A convolutional network with fully connected layer at the end
    Convolutions used: 
    - 3x3 Stepsize 1
    - Maxpool Stepsize 2
    - 3x3 Stepsize 1
    - Maxpool Stepsize 2
    - Flatten and fully connect from 7x7x32 to 10x1
    """
    # result display data
    DISPLAY_GROUP="CNN_EVAL"
    DISPLAY_GROUP_ENUM=4

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"
        
    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS))
        
        self.train_labels = tf.placeholder(tf.float32,
                                      shape=(
                                          None,
                                          self.NUM_LABELS))
        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS),
                               name="input")
        return self.x

    def initVariables(self):
        self.cw1 = tf.Variable(
            tf.truncated_normal([3,3,1,16],
                stddev=0.1,
                seed=42))
        self.cb1 = tf.Variable(tf.constant(0.1,shape=[16]))
        self.cw2 = tf.Variable(
            tf.truncated_normal([3,3,16,32],
                stddev=0.1,
                seed=42))
        self.cb2 = tf.Variable(tf.constant(0.1,shape=[32]))
        self.aw1 = tf.Variable(
            tf.truncated_normal([7*7*32,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.ab1 = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables = []
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        self.variables.append(self.aw1)
        self.variables.append(self.ab1)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding,use_cudnn_on_gpu=self.CUDNN)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = tf.nn.relu(self.conv2d(x,self.cw1,padding='SAME',strides=[1,1,1,1]) + self.cb1)
            self.convOut = self.maxpool2d(self.convOut)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw2,padding='SAME',strides=[1,1,1,1]) + self.cb2)
            self.convOut = self.maxpool2d(self.convOut)
            data_shape = self.convOut.get_shape().as_list()
            #Flatten
            reshaped = tf.reshape(self.convOut, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
            end = tf.matmul(reshaped,self.aw1) + self.ab1
        self.train_y = tf.nn.relu(end,name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))
        
class CNNUndercompleteAutoencoder(BenchmarkModel):
    """
    Autoencoders as used in GoogLeNet or Darknet19 (Used for YOLO object detection)
    Convolutions used: 
    - 3x3 Stepsize 2 - 16
    - 3x3 Stepsize 1 - 32
    - 1x1 Stepsize 1 - 16
    - 3x3 Stepsize 1 - 32
    - Flatten and fully connect from 7x7x32 to 10x1
    """
    # result display data
    DISPLAY_GROUP="CNN_EVAL"
    DISPLAY_GROUP_ENUM=4

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"
    
    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS))
        
        self.train_labels = tf.placeholder(tf.float32,
                                      shape=(
                                          None,
                                          self.NUM_LABELS))
        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS),
                               name="input")
        return self.x

    def initVariables(self):
        self.cw1 = tf.Variable(
            tf.truncated_normal([3,3,1,16],
                stddev=0.1,
                seed=42))
        self.cb1 = tf.Variable(tf.constant(0.1,shape=[16]))
        # Autoencoder Start
        self.cw2 = tf.Variable(
            tf.truncated_normal([3,3,16,32],
                stddev=0.1,
                seed=42))
        self.cb2 = tf.Variable(tf.constant(0.1,shape=[32]))
        
        self.cw3 = tf.Variable(
            tf.truncated_normal([1,1,32,16],
                stddev=0.1,
                seed=42))
        self.cb3 = tf.Variable(tf.constant(0.1,shape=[16]))

        self.cw4 = tf.Variable(
            tf.truncated_normal([3,3,16,32],
                stddev=0.1,
                seed=42))
        self.cb4 = tf.Variable(tf.constant(0.1,shape=[32]))
        # Autoencoder End
        self.aw1 = tf.Variable(
            tf.truncated_normal([7*7*32,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.ab1 = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables = []
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        self.variables.append(self.cw3)
        self.variables.append(self.cb3)
        self.variables.append(self.cw4)
        self.variables.append(self.cb4)
        self.variables.append(self.aw1)
        self.variables.append(self.ab1)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding,use_cudnn_on_gpu=self.CUDNN)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = tf.nn.relu(self.conv2d(x,self.cw1,padding='SAME',strides=[1,2,2,1]) + self.cb1)
            # Autoencoder Start
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw2,padding='SAME',strides=[1,1,1,1]) + self.cb2)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw3,padding='SAME',strides=[1,1,1,1]) + self.cb3)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw4,padding='SAME',strides=[1,1,1,1]) + self.cb4)
            # Autoencoder End
            self.convOut = self.maxpool2d(self.convOut)
        
            #Flatten
            data_shape = self.convOut.get_shape().as_list()
            reshaped = tf.reshape(self.convOut, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
            end = tf.matmul(reshaped,self.aw1) + self.ab1
        self.train_y = tf.nn.relu(end,name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))
        
class CNNOvercompleteAutoencoder(BenchmarkModel):
    """
    Overcomplete Autoencoder
    Convolutions used: 
    - 3x3 Stepsize 2 - 16
    - 3x3 Stepsize 1 - 32
    - 1x1 Stepsize 1 - 64
    - 3x3 Stepsize 1 - 32
    - Flatten and fully connect from 7x7x32 to 10x1
    """
    # result display data
    DISPLAY_GROUP="CNN_EVAL"
    DISPLAY_GROUP_ENUM=5

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    CUDNN=True
    DATASET="MNIST"
    
    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                     None,
                                     self.IMAGE_SIZE,
                                     self.IMAGE_SIZE,
                                     self.NUM_CHANNELS))
        
        self.train_labels = tf.placeholder(tf.float32,
                                      shape=(
                                          None,
                                          self.NUM_LABELS))
        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS),
                               name="input")
        return self.x

    def initVariables(self):
        self.cw1 = tf.Variable(
            tf.truncated_normal([3,3,1,16],
                stddev=0.1,
                seed=42))
        self.cb1 = tf.Variable(tf.constant(0.1,shape=[16]))
        # Autoencoder Start
        self.cw2 = tf.Variable(
            tf.truncated_normal([3,3,16,32],
                stddev=0.1,
                seed=42))
        self.cb2 = tf.Variable(tf.constant(0.1,shape=[32]))
        
        self.cw3 = tf.Variable(
            tf.truncated_normal([1,1,32,64],
                stddev=0.1,
                seed=42))
        self.cb3 = tf.Variable(tf.constant(0.1,shape=[64]))

        self.cw4 = tf.Variable(
            tf.truncated_normal([3,3,64,32],
                stddev=0.1,
                seed=42))
        self.cb4 = tf.Variable(tf.constant(0.1,shape=[32]))
        # Autoencoder End
        self.aw1 = tf.Variable(
            tf.truncated_normal([7*7*32,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.ab1 = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables = []
        self.variables.append(self.cw1)
        self.variables.append(self.cb1)
        self.variables.append(self.cw2)
        self.variables.append(self.cb2)
        self.variables.append(self.cw3)
        self.variables.append(self.cb3)
        self.variables.append(self.cw4)
        self.variables.append(self.cb4)
        self.variables.append(self.aw1)
        self.variables.append(self.ab1)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding,use_cudnn_on_gpu=self.CUDNN)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = tf.nn.relu(self.conv2d(x,self.cw1,padding='SAME',strides=[1,2,2,1]) + self.cb1)
            # Autoencoder Start
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw2,padding='SAME',strides=[1,1,1,1]) + self.cb2)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw3,padding='SAME',strides=[1,1,1,1]) + self.cb3)
            self.convOut = tf.nn.relu(self.conv2d(self.convOut,self.cw4,padding='SAME',strides=[1,1,1,1]) + self.cb4)
            # Autoencoder End
            self.convOut = self.maxpool2d(self.convOut)
            #Flatten
            data_shape = self.convOut.get_shape().as_list()
            reshaped = tf.reshape(self.convOut, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
            end = tf.matmul(reshaped,self.aw1) + self.ab1
        self.train_y = tf.nn.relu(end,name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))
