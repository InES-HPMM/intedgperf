from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy
import os
import time
import sys
from .BaseFramework import BaseFramework

DEBUG = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class TensorflowFramework(BaseFramework):

    def trainModel(self,benchModel,
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
        trainMult=4,
        silent=True,
        device="/device:GPU:0"):
        """
        Creates a session and trains it with the given model.
        """
        with tf.device(device):
            benchModel.initTrainPlaceholders()
            benchModel.initVariables()
            train_data_node = benchModel.getTrainInput()
            train_labels_node = benchModel.getTrainLabelVariable()
            y_train = benchModel.getTrainModel(train_data_node)
            loss = benchModel.getLoss(y_train)
            regularizers = benchModel.getRegularizer()


            loss += 5e-4 * regularizers

            batch = tf.Variable(0)
            train_size = train_labels.shape[0]

            learning_rate = tf.train.exponential_decay(
                0.01, # learning rate
                batch * benchModel.BATCH_SIZE, #index of batch
                train_size,
                0.95
            )
            if benchModel.TRAIN:
                # Use simple momentum for the optimization.
                optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)
                # Predictions for the minibatch, validation set and test set.
                train_prediction = tf.nn.softmax(y_train)
        
        # Create a new interactive session that we'll use in
        # subsequent code cells.
        if DEBUG:
            s = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        else:
            s = tf.InteractiveSession()

        
        # Use our newly created session as the default for 
        # subsequent operations.
        s.as_default()
        # Initialize all the variables we defined above.
        tf.global_variables_initializer().run()
        # Check if the Model is in a shape to be trained. Some models may be just the initialized values and no training done
        if not benchModel.TRAIN:
            return s

        # Validation & Test Data Preparation
        validation_dict = {train_data_node: benchModel.preprocessData(validation_data), 
                            train_labels_node: validation_labels}
        test_dict = {train_data_node: benchModel.preprocessData(test_data), 
                            train_labels_node: test_labels}

        # Train on all training data
        steps = int((train_size / benchModel.BATCH_SIZE) * trainMult)
        for step in range(steps):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * benchModel.BATCH_SIZE) % (train_size - benchModel.BATCH_SIZE)
            batch_data = train_data[offset:(offset + benchModel.BATCH_SIZE)]
            batch_labels = train_labels[offset:(offset + benchModel.BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.            
            feed_dict = {train_data_node: benchModel.preprocessData(batch_data),
                        train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            self.progressBar(step,steps)
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            # Print out the loss periodically.
            if not silent and step % 100 == 0:
                error, _ = self.errorRate(predictions, batch_labels)
                print('Step %d of %d' % (step, steps))
                print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
                val_predictions = s.run(train_prediction,feed_dict=validation_dict)
                print('Validation error: %.1f%%' % self.errorRate(
                    val_predictions, validation_labels)[0])
        print("\tEnd Training result:")
        test_predictions = s.run(train_prediction,feed_dict=test_dict)
        print('\tValidation error: %.1f%%' % self.errorRate(
                    test_predictions, test_labels)[0])
        return s

    def maybeTrain(self, data_dict, model_dict, work_dir="trained_models", testset=[], customCompiler=None, device="/device:GPU:0"):
        """ 
        Loads all models and trains them, should there not be an already trained model available
        """
        if customCompiler:
            customCompiler = customCompiler()
        classes = model_dict
        for cn,c in classes:
            if len(testset) > 0 and c.DISPLAY_GROUP not in testset:
                continue
            tf.reset_default_graph()
            # Make the results reproducable
            tf.set_random_seed(42)
            destination = work_dir+"/"+cn
            if(os.path.isfile(destination+".index")):
                print("Trained Model found: " + destination)
            else:
                print("Training: " + destination)
                modelUnderTest = c()
                s = self.trainModel(modelUnderTest,
                    data_dict[modelUnderTest.DATASET]["train_data"],
                    data_dict[modelUnderTest.DATASET]["train_labels"],
                    data_dict[modelUnderTest.DATASET]["validation_data"],
                    data_dict[modelUnderTest.DATASET]["validation_labels"],
                    data_dict[modelUnderTest.DATASET]["test_data"],
                    data_dict[modelUnderTest.DATASET]["test_labels"],
                    device=device
                    )
                graph = tf.Graph()
                graph.device(device)
                for variable in modelUnderTest.variables:
                    tf.contrib.copy_graph.copy_variable_to_graph(variable, graph)

                s.close()
                # Create the Export session, for that we create a non-default session
                sess = tf.InteractiveSession(graph=graph)
                sess.as_default()
                graph.as_default()
                x = modelUnderTest.initRunPlaceholders()
                variables = modelUnderTest.initVariables()
                y = modelUnderTest.getModel(x)
                tf.global_variables_initializer().run()
                writer = tf.summary.FileWriter(destination+"_tensorboard")
                writer.add_graph(sess.graph)
                saver = tf.train.Saver()
                saver.save(sess,destination)
                sess.close()
            if customCompiler:
                customCompiler.compile(destination)

    def measure_times(self,data_dict,model_dict,work_dir="trained_models",device="/device:GPU:0",testset=[],data_mult=3,customLoader=None,customRunner=None):
        """
        Measures the execution times for each benchmark case one input at the time, like it would be used in the field.
        """
        classes = model_dict
        testResults = {}
        for cn,c in classes:
            if len(testset) > 0 and c.DISPLAY_GROUP not in testset:
                continue
            tf.reset_default_graph()
            with tf.device(device):
                source = work_dir+"/"+cn
                print("Running: " + source)
                model = c()
                x = model.initRunPlaceholders()
                variables = model.initVariables()
                y = model.getModel(x)
            if DEBUG:
                s = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            else:
                s = tf.InteractiveSession()
            s.as_default()
            s.graph.device(device)
            tf.global_variables_initializer().run()
            if model.TRAIN:
                saver = tf.train.Saver()
                saver.restore(s,source)
                s.graph.finalize()
            data,_ = self.getData(model.DATASET,data_dict)
            mult_data = []
            for i in range(int(data_mult)):
                mult_data.extend(data)
            feedDicts = list(map(lambda e: { x: model.preprocessData(e) },mult_data))
            print("Start measurement...")
            tTot = 0
            for feedDict in feedDicts:
                # TODO: Do the custom data loader here if none available
                start = time.time()
                pred = s.run(y, feed_dict=feedDict)
                # TODO Call instead a custom runner, if any required
                # We actually dont care about the prediction yet
                # TODO: Maybe create a repo of predicitons to detect any interferences of optimizers
                end = time.time()
                tTot += end-start
                
            testResults[cn] = self.compileTestResult(cn, c, tTot, len(mult_data))
            s.close()
        return testResults

    def printSystemInformation(self):
        print("-----------------------------------------\nSystem Info\n-----------------------------------------")
        print("Running with Python Version {}.{}.{}".format(sys.version_info[0],sys.version_info[1],sys.version_info[2]))
        print(device_lib.list_local_devices())  # -- Gets stuck in the tensorflow/tensorflow:gpu-latest docker?
