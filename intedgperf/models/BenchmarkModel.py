try:
    import tensorflow as tf
    IS_DUMMY = False
except ImportError:
    IS_DUMMY = True
import numpy

class BenchmarkModel(object):
    """
    The abstract base class implmented by all different sorts of models.
    Defined in the model are inputs, outputs, and the model.
    """
    # result display data
    DISPLAY_GROUP="None"
    DISPLAY_GROUP_ENUM=99
    
    # Training control data
    TRAIN = 1

    # Test relevant data
    SEED=42
    DATASET="MNIST"

    BATCH_SIZE=60

    # Desired Datashape (AKA the x-shape)
    DATA_SHAPE=(1,28,28,1)
    
    def initTrainPlaceholders(self):
        self.train_x = tf.placeholder(tf.float32, shape=(1,1))
        self.train_labels=None

        return self.train_x,self.train_labels
        
    def initRunPlaceholders(self):
        self.x = tf.placeholder(tf.float32, shape=(1,1))
        return self.x

    def initVariables(self):
        self.variables = []
        return self.variables

    def getInput(self):
        """
        Returns the input placeholder variable for execution of the benchmark
        """
        return self.x
    def getTrainInput(self):
        """
        Returns the input placeholder variable for the training phase of the benchmark. 
        This is most likely using mini-batches to increase quality of the result and reduce training times.
        """
        return self.train_x
    def getModel(self,x):
        """Builds the model for inference running"""
        self.y = self.x
        return self.y
    def getTrainModel(self,x):
        """Builds the training model"""
        self.train_y = self.train_x
        return self.train_y
    def getTrainLabelVariable(self):
        return self.train_labels
    def getRegularizer(self):
        return 0
    def getLoss(self,train_y):
        return 1
    def preprocessData(self,data):
        if not len(self.DATA_SHAPE) == len(data.shape):
            data = numpy.reshape(data,self.DATA_SHAPE)
        return data