try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
    
from .BenchmarkModel import BenchmarkModel

class MovFC(BenchmarkModel):
    """
    A simple network consisting of flatten, a fully connected layer and a softmax operation
    """
    # result display data
    DISPLAY_GROUP="MOV_COMPT"
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
                                    self.NUM_CHANNELS),
                                    name="training_input")
        
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
        self.variables = []
        self.ann_weights = tf.Variable(
            tf.truncated_normal([
                self.IMAGE_SIZE*self.IMAGE_SIZE*self.NUM_CHANNELS, 
                self.NUM_LABELS],
                stddev=0.1,
                seed=42),
                name="ann_weights")
        self.ann_bias = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]),
        name="ann_bias")

        self.variables.append(self.ann_weights)
        self.variables.append(self.ann_bias)
        return self.variables

    def getTrainModel(self,x):
        with tf.name_scope("Training"):
            data_shape = x.get_shape().as_list()
            #Flatten
            reshaped = tf.reshape(x, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
            multOut = tf.matmul(reshaped,self.ann_weights)+self.ann_bias
        train_y = tf.nn.relu(multOut)
        return train_y

    def getModel(self,x):
        with tf.name_scope("Execution"):
            data_shape = x.get_shape().as_list()
            #Flatten
            reshaped = tf.reshape(x, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
            multOut = tf.matmul(reshaped,self.ann_weights)+self.ann_bias
        y = tf.nn.relu(multOut,name="output")
        return y
    
    def getRegularizer(self):
        return tf.nn.l2_loss(self.ann_weights) + tf.nn.l2_loss(self.ann_bias)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))