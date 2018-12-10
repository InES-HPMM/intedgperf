from ..LoaderBase import LoaderBase
from mvnc import mvncapi as mvnc
import os.path

class MovidiusLoader(LoaderBase):
    def load(self,trained_model_location="",all_data=[]):
        
        if not os.path.isfile(trained_model_location+".movidius.graph"):
            return None

        # Get the device
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('No Movidus devices found. Aborting')
            quit()
        device = mvnc.Device(devices[0])
        device.OpenDevice()


        # Load the Graph
        
        with open(trained_model_location+".movidius.graph", 'rb') as f:
            unallocatedGraph = f.read()
        
        graph = device.AllocateGraph(unallocatedGraph)
        userdata = {}
        userdata["graph"] = graph
        userdata["device"] = device
        userdata["unloader"] = self.unloadDevice
        return userdata


    def unloadDevice(self,userdata):
        userdata["graph"].DeallocateGraph()
        userdata["device"].CloseDevice()