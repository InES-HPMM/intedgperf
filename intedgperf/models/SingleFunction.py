try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
    
from .BenchmarkModel import BenchmarkModel

class TransferMeasurement(BenchmarkModel):
    """
    The point of this test is to measure how long it takes to move the data over to the calculating device and do a trivial task. 
    This way it will be possible to gage the 
    """
    DISPLAY_GROUP="SINGLE_FUNC_T"
    DISPLAY_GROUP_ENUM=0
    TRAIN = 0
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10

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

    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                           shape=(
                               None,
                               self.IMAGE_SIZE,
                               self.IMAGE_SIZE,
                               self.NUM_CHANNELS),name="input")
        return self.x

    def initVariables(self):
        self.bias = tf.Variable(tf.constant(0.1,shape=[
                                self.IMAGE_SIZE,
                                self.IMAGE_SIZE,
                                self.NUM_CHANNELS]))
        self.variables=[]
        self.variables.append(self.bias)

    def getTrainModel(self,x):
        data_shape = x.get_shape().as_list()
        #Flatten
        train_y = tf.add(x, self.bias, name="output")
        return train_y

    def getModel(self,x):

        return self.getTrainModel(x)
        
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))
        
class OnlyFC(BenchmarkModel):
    """
    A simple network consisting of flatten, a fully connected layer and a softmax operation
    """
    # result display data
    DISPLAY_GROUP="SINGLE_FUNC"
    DISPLAY_GROUP_ENUM=1

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
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
                               self.NUM_CHANNELS),name="input")
        return self.x

    def initVariables(self):
        self.variables = []
        self.ann_weights = tf.Variable(
            tf.truncated_normal([
                self.IMAGE_SIZE*self.IMAGE_SIZE*self.NUM_CHANNELS, 
                self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.ann_bias = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables.append(self.ann_weights)
        self.variables.append(self.ann_bias)
        return self.variables


    def getTrainModel(self,x):
        data_shape = x.get_shape().as_list()
        #Flatten
        reshaped = tf.reshape(x, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
        train_y = tf.nn.relu(tf.matmul(reshaped,self.ann_weights)+self.ann_bias,name="output")
        return train_y

    def getModel(self,x):

        return self.getTrainModel(x)
    
    def getRegularizer(self):
        return tf.nn.l2_loss(self.ann_weights) + tf.nn.l2_loss(self.ann_bias)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))
        
class OnlyCNN(BenchmarkModel):
    """
    A simple network consisting of flatten, a fully connected layer and a softmax operation
    """
    # result display data
    DISPLAY_GROUP="SINGLE_FUNC"
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
                               self.NUM_CHANNELS),name="input")
        return self.x

    def initVariables(self):
        self.variables = []
        self.cnn_weights = tf.Variable(
            tf.truncated_normal([self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.cnn_bias = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables.append(self.cnn_weights)
        self.variables.append(self.cnn_bias)
        return self.variables

    def conv2d(self,x,W,stride=1,padding='SAME'):
        return tf.nn.conv2d(x,W,[1,stride,stride,1],padding,use_cudnn_on_gpu=self.CUDNN)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        self.convOut = tf.nn.relu(self.conv2d(x,self.cnn_weights,padding='VALID') + self.cnn_bias)
        #Flatten
        self.train_y = tf.reshape(self.convOut, [-1,self.NUM_LABELS],name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

class OnlyMaxpool(BenchmarkModel):
    """
    A simple network consisting of flatten, a fully connected layer and a softmax operation
    """
    # result display data
    DISPLAY_GROUP="SINGLE_FUNC"
    DISPLAY_GROUP_ENUM=3

    # Training control data
    TRAIN = 0

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
                               self.NUM_CHANNELS),name="input")
        return self.x

    def initVariables(self):
        self.variables = []
        self.cnn_weights = tf.Variable(
            tf.truncated_normal([self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.cnn_bias = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables.append(self.cnn_weights)
        self.variables.append(self.cnn_bias)
        return self.variables

    def conv2d(self,x,W,stride=1,padding='SAME'):
        return tf.nn.conv2d(x,W,[1,stride,stride,1],padding,use_cudnn_on_gpu=self.CUDNN)
        
    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        self.train_y = self.maxpool2d(x)
        return self.train_y
    
    def getModel(self,x):
        return tf.identity(self.getTrainModel(x),name="output")
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.maxpool2d(self.train_x),logits=train_y))

class OnlyDeconv(BenchmarkModel):
    """
    A simple network consisting of flatten, a fully connected layer and a softmax operation
    """
    # result display data
    DISPLAY_GROUP="SINGLE_FUNC"
    DISPLAY_GROUP_ENUM=4

    # Training control data
    TRAIN = 0

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
        self.dcnn_weights = tf.Variable(
            tf.truncated_normal([self.IMAGE_SIZE,self.IMAGE_SIZE,1,1],
                stddev=0.1,
                seed=42))
        self.variables = []
        self.variables.append(self.dcnn_weights)
        return self.variables

    def deconv2d(self,x,W,stride=1,padding='SAME'):
        shape = x.get_shape().as_list()
        output_shape = [1,self.IMAGE_SIZE*2,self.IMAGE_SIZE*2,shape[-1]]
        return tf.nn.conv2d_transpose(x,W,output_shape,[1,2,2,1],padding=padding)

    def maxpool2d(self,x,ksize=2,stride=2,padding='SAME'):
        return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)

    def getTrainModel(self,x):
        self.train_y = self.deconv2d(x,self.dcnn_weights)
        return self.train_y
    
    def getModel(self,x):
        return tf.identity(self.getTrainModel(x),name="output")
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_x,logits=self.maxpool2d(self.train_y)))

class SimpleLSTM(BenchmarkModel):
    DISPLAY_GROUP="SINGLE_FUNC"
    DISPLAY_GROUP_ENUM=5

    # Test relevant data
    DATASET="SEQ"
    SERIES_LENGTH = 10
    NUM_HIDDEN = 12
    BATCH_SIZE=100

    DATA_SHAPE=(1,SERIES_LENGTH,1)

    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                    None,
                                    self.SERIES_LENGTH,
                                    1
                                )
                            )
        
        self.train_labels = tf.placeholder(
                                    tf.float32,
                                    shape=(
                                          None,
                                          self.SERIES_LENGTH+1)
                                    )

        return self.train_x, self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                                shape=(
                                    None,
                                    self.SERIES_LENGTH,
                                    1
                                )
                               ,name="input"
                            )
        return self.x

    def initVariables(self):
        self.w = tf.Variable(
        tf.truncated_normal(
                [
                    self.NUM_HIDDEN,
                    self.SERIES_LENGTH+1
                ]
            )
        )
        self.b = tf.Variable(tf.constant(0.1, shape=[self.SERIES_LENGTH+1]))
        self.variables = []
        self.variables.append(self.w)
        self.variables.append(self.b)
        return self.variables

    def getTrainModel(self,x):
        # RNN
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)
        val,status = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        val = tf.transpose(val,[1,0,2])
        last = tf.gather(val,int(val.get_shape()[0])-1)

        # Fully Connected
        prediction = tf.nn.softmax(tf.matmul(last,self.w)+self.b)

        return prediction
    def getModel(self,x):
        # RNN
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)
        val,status = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        val = tf.transpose(val,[1,0,2])
        last = tf.gather(val,int(val.get_shape()[0])-1)

        # Fully Connected
        prediction = tf.nn.softmax(tf.matmul(last,self.w)+self.b,name="output")
        return prediction

    def getLoss(self,train_y):
        return -tf.reduce_sum(self.train_labels * tf.log(tf.clip_by_value(train_y,1e-10,1.0)))

class SimpleGRU(BenchmarkModel):
    DISPLAY_GROUP="SINGLE_FUNC"
    DISPLAY_GROUP_ENUM=6

    # Test relevant data
    DATASET="SEQ"
    SERIES_LENGTH = 10
    NUM_HIDDEN = 12
    BATCH_SIZE=100

    DATA_SHAPE=(1,SERIES_LENGTH,1)

    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32,
                                 shape=(
                                    None,
                                    self.SERIES_LENGTH,
                                    1
                                )
                            )
        
        self.train_labels = tf.placeholder(
                                    tf.float32,
                                    shape=(
                                          None,
                                          self.SERIES_LENGTH+1)
                                    )

        return self.train_x, self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32,
                                shape=(
                                    None,
                                    self.SERIES_LENGTH,
                                    1
                                )
                               ,name="input"
                            )
        return self.x

    def initVariables(self):
        self.w = tf.Variable(
        tf.truncated_normal(
                [
                    self.NUM_HIDDEN,
                    self.SERIES_LENGTH+1
                ]
            )
        )
        self.b = tf.Variable(tf.constant(0.1, shape=[self.SERIES_LENGTH+1]))
        self.variables = []
        self.variables.append(self.w)
        self.variables.append(self.b)
        return self.variables

    def getTrainModel(self,x):
        # RNN
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)
        val,status = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        val = tf.transpose(val,[1,0,2])
        last = tf.gather(val,int(val.get_shape()[0])-1)

        # Fully Connected
        prediction = tf.nn.softmax(tf.matmul(last,self.w)+self.b)

        return prediction
    def getModel(self,x):
        # RNN
        cell = tf.nn.rnn_cell.GRUCell(self.NUM_HIDDEN)
        val,status = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
        val = tf.transpose(val,[1,0,2])
        last = tf.gather(val,int(val.get_shape()[0])-1)

        # Fully Connected
        prediction = tf.nn.softmax(tf.matmul(last,self.w)+self.b,name="output")
        return prediction

    def getLoss(self,train_y):
        return -tf.reduce_sum(self.train_labels * tf.log(tf.clip_by_value(train_y,1e-10,1.0)))