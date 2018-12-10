from ..CompilerBase import CompilerBase
import subprocess
import os

class MovidiusCompiler(CompilerBase):
    def compile(self,trained_model_location,model_input="input",model_output="output"):
        userdata={}
        print("Movidius compiler used for model " + trained_model_location)
        #../benchmark_model/CNNOvercompleteAutoencoder
        # mvNCCompile mnist_inference.meta -s 12 -in input -on output -o mnist_inference.graph
        command = ["mvNCCompile",trained_model_location+".meta","-in", model_input, "-on", model_output, "-o",trained_model_location+".movidius.graph"]
        print("Executing command\n")
        print(command)
        subprocess.call(command)
        return userdata