from ..RunnerBase import RunnerBase
from mvnc import mvncapi as mvnc


class MovidiusRunner(RunnerBase):
    def run(self,trained_model_location="",data=None,userdata={}):
        userdata['graph'].LoadTensor(data,None)
        result = userdata['graph'].GetResult()
        return result