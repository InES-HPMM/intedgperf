#!/usr/bin/python
from __future__ import print_function
import base64
import os

import numpy
import importlib,inspect,sys
import time 
import json
import argparse
import pkgutil
try:
    set
except NameError:
    from sets import Set as set

import imp

import intedgperf.DataPreparer as DataPreparer


from intedgperf.BaseFramework import BaseFramework

from intedgperf.module.CompilerBase import CompilerBase
from intedgperf.module.LoaderBase import LoaderBase
from intedgperf.module.RunnerBase import RunnerBase

MODEL_DIRECTORY = "intedgperf/models/"
MODULE_DIRECTORY = "intedgperf/module/"
DEBUG = False

def createModelDict():
    """Creates a dictionary with all classes located in the models directory.
      Only classes inheriting from the BenchmarkModel abstract class are added."""
    models = filter(lambda x: x.endswith(".py"), os.listdir(MODEL_DIRECTORY))
    classes = {}
    for m in models:
        modelName = m.split('.')[0]
        if modelName.startswith("_"):
            continue
        modelLocation = MODEL_DIRECTORY+modelName
        mod = importlib.import_module(modelLocation.replace("/","."))
        for cname, obj in inspect.getmembers(mod, inspect.isclass):
            if cname is not 'BenchmarkModel':
                classes[cname] = obj
    return classes

def createModuleDict():
    """Load all hardware specific compilers,loaders and runners, located in subfolders in the module directory"""
    moduleDict = {}
    moduleDirs = filter(lambda x: os.path.isdir(MODULE_DIRECTORY+x), os.listdir(MODULE_DIRECTORY))

    for module in moduleDirs:
        moduleFiles = filter(lambda x: x.endswith(".py") and not x.startswith("__"), 
            os.listdir(MODULE_DIRECTORY+module))
        for moduleFile in moduleFiles:
            try:
                mod = importlib.import_module(MODULE_DIRECTORY.replace("/",".")+module+"."+moduleFile.split(".")[0])
                for cname, obj in inspect.getmembers(mod,inspect.isclass):
                    if issubclass(obj,CompilerBase) or issubclass(obj,LoaderBase) or issubclass(obj,RunnerBase):
                        moduleDict[cname] = obj
            except ImportError:
                print("Could not load Module {}, missing library".format(moduleFile))
    moduleDict.pop("CompilerBase",None)
    moduleDict.pop("LoaderBase",None)
    moduleDict.pop("RunnerBase",None)
    return moduleDict
    
def fillDataDict(dataset,packed_data,data_dict={}):
    """Creates a dictrionary with datasources split in the respective parts"""
    data_dict[dataset] = {}
    data_dict[dataset]["train_data"],data_dict[dataset]["train_labels"],data_dict[dataset]["validation_data"],data_dict[dataset]["validation_labels"],data_dict[dataset]["test_data"],data_dict[dataset]["test_labels"] = packed_data
    return data_dict

def getSubclasses(classDictionary,superclass):
    return list(filter(lambda c: issubclass(c,superclass),classDictionary.values()))


if __name__ == "__main__":
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nIssues/Warnings for environment above (Tensorflow version etc)")    
    print("#########################################\nBeginning of Programm\n#########################################")
    parser = argparse.ArgumentParser(description='''Benchmark for machine learning operations to be used with a variety of hardwares. Combine resutls using CombineResults.py''',
    epilog="Example: "+str(sys.argv[0])+" -d '/gpu:0' -n 'TX2'")
    parser.add_argument("-ls",action='store_true' ,help="Lists all possible testsets")
    parser.add_argument("-s","--testset",type=str,default='',nargs='*',help="The testset to run")
    parser.add_argument("-d","--device",type=str,default="/cpu:0",help="Device the test is to be run on, most likely /cpu:0 or /gpu:0")
    parser.add_argument("-n","--name",type=str,default="Unnamed Test",help="Display name for the results")
    parser.add_argument("-m","--multiplier",type=float,default="1",help="Run data multiplier. The test will run all the data multiplier times. A higher multiplier means a longer running time, but a more accurate average")
    parser.add_argument("--trainOnly",action='store_true',help="Only performs the training of the models or executes the custom compiler, then ends")
    parser.add_argument("--runOnly",action='store_true',help="Only runs the performance benchmark")
    parser.add_argument("--compiler",type=str,help="Compiler module for custom hardware. When provided will be run in any case. With the models.")
    parser.add_argument("--loader",type=str,help="Loader defining how data for the custom hardware needs to be prepared")
    parser.add_argument("--runner",type=str,help="Module for running the hardware")
    args = parser.parse_args()
    
    
    print("-----------------------------------------\nImport Framework\n-----------------------------------------")
    framework = BaseFramework()
    try:
        from intedgperf.TensorflowFramework import TensorflowFramework
        print("Using Tensorflow as Backend")
        framework = TensorflowFramework()
    except ImportError:
        print("No Tensorflow found")

    framework.printSystemInformation()

    if args.ls:
        print("-----------------------------------------\nTestsets\n-----------------------------------------")
        classes = createModelDict()
        lis = set()
        for cn,c in classes.items():
            lis.add(c.DISPLAY_GROUP)
        for l in lis:
            print(l)
        print("-----------------------------------------")
        exit()
    
    

    print("-----------------------------------------")
    device = args.device
    testname = args.name
    testsets = args.testset
    modules = createModuleDict()
    customCompiler = None
    customLoader = None
    customRunner = None

    ## Process possible modules
    if args.compiler:
        # Compiler defined by user
        if args.compiler not in modules:
            # Defined compiler does not exist
            print("Compiler " + args.compiler + " not found!")
            print("Compiler classes found: ")
            for c,_ in getSubclasses(modules,CompilerBase):
                print("\t- "+c)
            exit()
        customCompiler = modules[args.compiler]
        if not issubclass(customCompiler,CompilerBase):
            print("Error, " + customCompiler + " does not inherit from CompilerBase")
            exit()
        print(customCompiler.__name__,"loaded")
    if args.loader:
        # loader defined by user
        if args.loader not in modules:
            # Defined loader does not exist
            print("Data Loader " + args.loader + " not found!")
            print("Data Loader classes found: ")
            for c,_ in getSubclasses(modules,LoaderBase):
                print("\t- "+c)
            exit()
        customLoader = modules[args.loader]
        if not issubclass(customLoader,LoaderBase):
            print("Error, " + customLoader + " does not inherit from LoaderBase")
            exit()
        print(customLoader.__name__,"loaded")
    if args.runner:
        # Runner defined by user
        if args.runner not in modules:
            # Defined runner does not exist
            print("Runner " + args.runner + " not found!")
            print("Runner classes found: ")
            for c,_ in getSubclasses(modules,RunnerBase):
                print("\t- "+c)
            exit()
        customRunner = modules[args.runner]
        if not issubclass(customRunner,RunnerBase):
            print("Error, " + customRunner + " does not inherit from RunnerBase")
            exit()
        print(customRunner.__name__,"loaded")

    # Load data

    print("-----------------------------------------\nLoading data\n-----------------------------------------")
    data_dict = fillDataDict("MNIST",DataPreparer.downloadMNIST())
    data_dict = fillDataDict("SEQ",DataPreparer.binaryCount())

    classes = sorted(createModelDict().items(),key=lambda v: (v[1].DISPLAY_GROUP,v[1].DISPLAY_GROUP_ENUM))
    if not args.runOnly:
        print("-----------------------------------------\nTraining new models\n-----------------------------------------")
        # train, if not already trained    
        framework.maybeTrain(data_dict,classes,testset=testsets,customCompiler=customCompiler,device=device)
        print("-----------------------------------------\nRunning benchmark\n-----------------------------------------")
        # Getting a higher process-priority
        try:
            print("Running tests with niceness: " + str(os.nice(-20)))
        except OSError:
            print("Can not alter niceness, running with default niceness")
    
    # run the test with GPU & CPU
    benchmark_result = {}
    benchmark_result[testname] = framework.measure_times(data_dict,classes,device=device,testset=testsets,data_mult=args.multiplier,customLoader=customLoader,customRunner=customRunner)
    print("-----------------------------------------\nSaving results\n-----------------------------------------")
    outFile = testname+'.json'
    with open(outFile,'w') as f:
        f.write(json.dumps(benchmark_result))
        print("File " + outFile + " written.")
