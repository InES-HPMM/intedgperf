try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
    
from .BenchmarkModel import BenchmarkModel

######################## 1 Fully Connected layers, having an activation function #########################

class ANN1NoActivation(BenchmarkModel):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with no activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP="ACTIVATION_FC1"
    DISPLAY_GROUP_ENUM=1

    # Test relevant data
    BATCH_SIZE=12
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    HIDDEN_LAYER=512
    NUM_LABELS=10
    DATASET="MNIST"
    SEED = 42
    STDDEV=0.1
    BIAS=0.1

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
        self.variables = []
        self.fc1 = tf.Variable(
            tf.truncated_normal([
                self.IMAGE_SIZE*self.IMAGE_SIZE*self.NUM_CHANNELS, 
                self.NUM_LABELS],
                stddev=self.STDDEV,
                seed=self.SEED))
        self.fc1_b = tf.Variable(tf.constant(self.BIAS,shape=[self.NUM_LABELS]))
        self.variables.append(self.fc1)
        self.variables.append(self.fc1_b)
        return self.variables
        
    def getTrainModel(self,x):
        return self.getModel(x)

    def getModel(self,x):
        with tf.name_scope("model"):
            data_shape = x.get_shape().as_list()
            #Flatten
            reshaped = tf.reshape(x, [-1,data_shape[1]*data_shape[2]*data_shape[3]])
            out = self.getOperations(reshaped)
        return tf.identity(out,name="output")

    def getOperations(self,reshapedData):
        return tf.matmul(reshapedData,self.fc1)+self.fc1_b

    def getRegularizer(self):
        return tf.nn.l2_loss(self.fc1) + tf.nn.l2_loss(self.fc1_b)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

class ANN1SigmoidActivation(ANN1NoActivation):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with a sigmoid activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP_ENUM=2


    def getOperations(self,reshapedData):
        return tf.sigmoid(tf.matmul(reshapedData,self.fc1)+self.fc1_b)

class ANN1TanHActivation(ANN1NoActivation):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with a TanH activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP_ENUM=3

    def getOperations(self,reshapedData):
        return tf.tanh(tf.matmul(reshapedData,self.fc1)+self.fc1_b)

class ANN1ReLUActivation(ANN1NoActivation):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with a ReLU activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP_ENUM=4

    def getOperations(self,reshapedData):
        return tf.nn.relu(tf.matmul(reshapedData,self.fc1)+self.fc1_b)

class ANN1leakyReLUActivation(ANN1NoActivation):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with a leaky ReLU function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP_ENUM=5

    # Test relevant data
    ALPHA=0.2

    def getOperations(self,reshapedData):
        return tf.nn.leaky_relu(tf.matmul(reshapedData,self.fc1)+self.fc1_b, alpha=self.ALPHA)

class ANN1ELUActivation(ANN1NoActivation):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with ann ELU activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP_ENUM=6

    # Test relevant data
    def getOperations(self,reshapedData):
        return tf.nn.elu(tf.matmul(reshapedData,self.fc1)+self.fc1_b)

class ANN1SELUActivation(ANN1NoActivation):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with an SELU activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP_ENUM=7

    def getOperations(self,reshapedData):
        return tf.nn.selu(tf.matmul(reshapedData,self.fc1)+self.fc1_b)

######################## 1 Convolution layers, having an activation function #########################

class CNN1NoActivation(BenchmarkModel):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with no activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP="ACTIVATION_CNN1"
    DISPLAY_GROUP_ENUM=1

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    DATASET="MNIST"
    SEED = 42
    STDDEV=0.1
    BIAS=0.1

        

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
        self.variables = []
        self.cnn_weights = tf.Variable(
            tf.truncated_normal([self.IMAGE_SIZE,self.IMAGE_SIZE,self.NUM_CHANNELS,self.NUM_LABELS],
                stddev=0.1,
                seed=42))
        self.cnn_bias = tf.Variable(tf.constant(0.1,shape=[self.NUM_LABELS]))
        self.variables.append(self.cnn_weights)
        self.variables.append(self.cnn_bias)
        return self.variables

    def conv2d(self,x,W,strides=[1,1,1,1],padding='SAME'):
        return tf.nn.conv2d(x,W,strides,padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = self.activationFunction(self.conv2d(x,self.cnn_weights,padding='VALID') + self.cnn_bias)
            data_shape = self.convOut.get_shape().as_list()
        #Flatten
        self.train_y = tf.reshape(self.convOut, [-1,self.NUM_LABELS],name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

    def activationFunction(self,x):
        return x

class CNN1SigmoidActivation(CNN1NoActivation):
    DISPLAY_GROUP_ENUM=2
    def activationFunction(self,x):
        return tf.sigmoid(x)

class CNN1TanHActivation(CNN1NoActivation):
    DISPLAY_GROUP_ENUM=3
    def activationFunction(self,x):
        return tf.tanh(x)

class CNN1ReLuActivation(CNN1NoActivation):
    DISPLAY_GROUP_ENUM=4
    def activationFunction(self,x):
        return tf.nn.relu(x)

class CNN1leakyReLuActivation(CNN1NoActivation):
    DISPLAY_GROUP_ENUM=5
    ALPHA=0.2
    def activationFunction(self,x):
        return tf.nn.leaky_relu(x,alpha=self.ALPHA)

class CNN1ELUActivation(CNN1NoActivation):
    DISPLAY_GROUP_ENUM=6
    def activationFunction(self,x):
        return tf.nn.elu(x)

class CNN1SELUActivation(CNN1NoActivation):
    DISPLAY_GROUP_ENUM=7
    def activationFunction(self,x):
        return tf.nn.selu(x)

######################## 3 Convolution layers, each having an activation function #########################

class CNN3NoActivation(BenchmarkModel):
    """
    A network consisting of flattening the input image 28x28x1 => 784x1 vector
    Then a fully connected layer to 512x1 with no activation function and then to 10x1 with softmax operation
    """
    # result display data
    DISPLAY_GROUP="ACTIVATION_CNN3"
    DISPLAY_GROUP_ENUM=1

    # Test relevant data
    BATCH_SIZE=60
    NUM_CHANNELS=1
    IMAGE_SIZE=28
    NUM_LABELS=10
    DATASET="MNIST"
    SEED = 42
    STDDEV=0.1
    BIAS=0.1

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
        return tf.nn.conv2d(x,W,strides,padding)

    def getTrainModel(self,x):
        with tf.name_scope("model"):
            self.convOut = self.activationFunction(self.conv2d(x,self.cw1,padding='SAME',strides=[1,2,2,1]) + self.cb1)
            self.convOut = self.activationFunction(self.conv2d(self.convOut,self.cw2,padding='SAME',strides=[1,2,2,1]) + self.cb2)
            self.convOut = self.activationFunction(self.conv2d(self.convOut,self.cw3,padding='VALID') + self.cb3)
        #Flatten
        self.train_y = tf.reshape(self.convOut, [-1,self.NUM_LABELS],name="output")
        return self.train_y
    
    def getModel(self,x):
        return self.getTrainModel(x)
    
    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))

    def activationFunction(self,x):
        return x

class CNN3SigmoidActivation(CNN3NoActivation):
    DISPLAY_GROUP_ENUM=2
    def activationFunction(self,x):
        return tf.sigmoid(x)

class CNN3TanHActivation(CNN3NoActivation):
    DISPLAY_GROUP_ENUM=3
    def activationFunction(self,x):
        return tf.tanh(x)

class CNN3ReLuActivation(CNN3NoActivation):
    DISPLAY_GROUP_ENUM=4
    def activationFunction(self,x):
        return tf.nn.relu(x)

class CNN3leakyReLuActivation(CNN3NoActivation):
    DISPLAY_GROUP_ENUM=5
    ALPHA=0.2
    def activationFunction(self,x):
        return tf.nn.leaky_relu(x,alpha=self.ALPHA)

class CNN3ELUActivation(CNN3NoActivation):
    DISPLAY_GROUP_ENUM=6
    def activationFunction(self,x):
        return tf.nn.elu(x)

class CNN3SELUActivation(CNN3NoActivation):
    DISPLAY_GROUP_ENUM=7
    def activationFunction(self,x):
        return tf.nn.selu(x)
