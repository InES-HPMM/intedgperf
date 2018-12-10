try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
    
from .BenchmarkModel import BenchmarkModel

class TensorflowTutorial(BenchmarkModel):
    """
    The Model as seen in Tensorflow tutorial mixed with the old tensorflow structure
    https://www.tensorflow.org/tutorials/

    https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    """
    # result display data
    DISPLAY_GROUP="KERAS_MODELS"
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

    def getTrainModel(self,x):
        y = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Dense(512,activation=tf.nn.relu)(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        self.train_y = tf.keras.layers.Dense(self.NUM_LABELS,activation=tf.nn.softmax)(y)
        return self.train_y

    def getModel(self,x):
        y = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Dense(512,activation=tf.nn.relu)(y)
        self.y = tf.keras.layers.Dense(self.NUM_LABELS,activation=tf.nn.softmax,name="output")(y)
        return self.y

    def getLoss(self,train_y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels,logits=train_y))