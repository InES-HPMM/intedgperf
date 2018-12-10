# Benchmark
Use docker if possible. https://hub.docker.com/r/tensorflow/tensorflow/

## Running the docker
This is only relevant if you're using docker. The commands are to be run from the root directory of the git project.


###GPU
```
nvidia-docker run -it --cap-add=sys_nice -v $(pwd)/:/notebooks/host tensorflow/tensorflow:latest-gpu-py3 bash
```


###CPU
```
docker run -it --cap-add=sys_nice -v $(pwd):/notebooks/host tensorflow/tensorflow:latest-py3 bash
```


## Running a test
Exeuting other things on the system under test will yield in worse results. Thus it's recommended to limit programs executed on the test system to a minimum.
```
python Run.py -n "QuadroK620_5" -d "/gpu:0" -s SINGLE_FUNC
```



### Source for TX1 & 2 Tensorflow and Install instructions
1. Setup using Jetpack JetPack-L4T-3.1-linux-x64.run  (https://developer.nvidia.com/embedded/jetpack-archive)
2. Aquire whl files https://github.com/jetsonhacks/installTensorFlowJetsonTX (cp27 means it's for python 2.7, cp35 is python 3.5)
3. Generate Locale: 
    ```
    sudo locale-gen en_US.UTF-8
    export LC_ALL="en_US.UTF-8"
    ```
4. `pip install tensorflow-1.3.0-cp27-cp27mu-linux_aarch64.whl`
