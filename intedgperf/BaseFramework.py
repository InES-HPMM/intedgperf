import numpy
import time
import sys
class BaseFramework():

    def errorRate(self,predictions, labels):
        """Return the error rate and confusions."""
        correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
        total = predictions.shape[0]

        error = 100.0 - (100 * float(correct) / float(total))

        confusions = numpy.zeros([10, 10], numpy.float32)
        bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(labels, 1))
        for predicted, actual in bundled:
            confusions[predicted, actual] += 1
            
        return error, confusions
    
    def trainModel(self,benchModel,
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
        trainMult=4,
        silent=True,
        device="/gpu:0"):
        print("BaseFramework trainModel not implemented yet!")

    def maybeTrain(self,data_dict,model_dict,work_dir="trained_models",testset=[],customCompiler=None,device=None):
        classes = model_dict
        if customCompiler:
            customCompiler = customCompiler()
            for cn,c in classes:
                if len(testset) > 0 and c.DISPLAY_GROUP not in testset:
                    continue
                destination = work_dir+"/"+cn
                customCompiler.compile(destination)
            
    
    def getData(self,dataset,data_dict):
        """Creates a list with all available data for the testrun"""
        allData = []
        allData.extend(data_dict[dataset]["train_data"])
        allData.extend(data_dict[dataset]["validation_data"])
        allData.extend(data_dict[dataset]["test_data"])
        allLabel = []
        allLabel.extend(data_dict[dataset]["train_labels"])
        allLabel.extend(data_dict[dataset]["validation_labels"])
        allLabel.extend(data_dict[dataset]["test_labels"])
        return allData,allLabel

    def measure_times(self,data_dict,model_dict,work_dir="trained_models",device="/gpu:0",testset=[],data_mult=3,customLoader=None,customRunner=None):
        """
        Measures the execution times for each benchmark case one input at the time, like it would be used in the field.
        """
        testResults={}
        if customLoader:
            customLoader = customLoader()
        else:
            print("No loader found")
        if customRunner:
            customRunner = customRunner()
        else:
            print("No runner found - Aborting")
            exit()
        
        for cn,c in model_dict:
            if len(testset) > 0 and c.DISPLAY_GROUP not in testset:
                continue
            print("**************************")
            print("Processing {}".format(cn))
            print("**************************")
            data,_ = self.getData(c.DATASET,data_dict)
            mult_data = []
            if data_mult >= 1:
                for i in range(int(data_mult)):
                    mult_data.extend(data)
            else:
                mult_data = data[:int(len(data)*data_mult)]
            saved_model = work_dir+"/"+cn
            userdata={}
            if customLoader:
                userdata = customLoader.load(trained_model_location=saved_model,all_data=mult_data)
            if customRunner and userdata:
                print("Running " + saved_model)
                lastInfo = time.clock()
                tTot = 0
                for i,d in enumerate(mult_data):
                    if time.clock()-lastInfo > 0.5:
                        self.progressBar(i,len(mult_data))
                        lastInfo = time.clock()
                    start = time.clock()
                    customRunner.run(userdata=userdata,data=d)
                    end = time.clock()
                    tTot += end-start
            else:
                continue

            if "unloader" in userdata:
                userdata["unloader"](userdata)

            testResults[cn] = self.compileTestResult(cn, c, tTot, len(mult_data))
        return testResults
            
            

    def compileTestResult(self, testClassName, testClass, totalTime, dataLength):
            tAvg = (totalTime / dataLength)*1000
            print("Time for " + testClassName + " " + str(totalTime) + " seconds in total, " + str(tAvg) + "ms on average per record (" + str(dataLength)+ " records )")
            runResults = {}
            runResults["time_total"] = totalTime
            runResults["time_average"] = tAvg
            runResults["run_test_number"] = dataLength
            runResults["group"] = testClass.DISPLAY_GROUP
            runResults["group_enum"] = testClass.DISPLAY_GROUP_ENUM
            return runResults

    def printSystemInformation(self):
        print("BaseFramework printSystemInformation not implemented yet!")

    def progressBar(self,value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()