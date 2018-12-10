from __future__ import print_function
import os
import numpy
import gzip
import urllib
import random

def maybe_download(filename, source_url, work_dir):
    """A helper to download the data files if not present."""
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    filepath = os.path.join(work_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(source_url + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath

def extract_data(filename, num_images,img_size=28,pixel_depth=255):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
  
    For MNIST data, the number of channels is always 1.

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    ----

    We can get the total num_images by:
    with gzip.open(test_data_filename) as f:
    # Print the header fields.
    for field in ['magic number', 'image count', 'rows', 'columns']:
        # struct.unpack reads the binary data provided by f.read.
        # The format string '>i' decodes a big-endian integer, which
        # is the encoding of the data.
        print(field, struct.unpack('>i', f.read(4))[0])
    """
    with gzip.open(filename) as bytestream:
    # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)
        buf = bytestream.read(img_size*img_size * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (pixel_depth / 2.0)) / pixel_depth # Scale the data from -1 to 1
        data = data.reshape(num_images, img_size, img_size, 1)
        return data

def extract_labels(filename,num_images,entry_size_bytes=1,num_labels=10):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(entry_size_bytes * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(num_labels) == labels[:, None]).astype(numpy.float32)
    
def splitData(train_data,train_labels,validation_size=5000):
    return  (train_data[:validation_size, :, :, :],
            train_labels[:validation_size],
            train_data[validation_size:, :, :, :],
            train_labels[validation_size:])

def downloadMNIST(work_dir="data/mnist-data",source_url='https://storage.googleapis.com/cvdf-datasets/mnist/'):
    """
    Downloads the MNIST dataset into workdir
    work_dir: default /tmp/mnist-data
    source_url: default 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        alternal source url: 'http://yann.lecun.com/exdb/mnist/'
    """
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz',source_url,work_dir)
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz',source_url,work_dir)
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz',source_url,work_dir)
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz',source_url,work_dir)
    
    train_set_size = 60000
    test_set_size = 10000
    train_data = extract_data(train_data_filename,train_set_size)
    test_data = extract_data(test_data_filename,test_set_size)
    train_labels = extract_labels(train_labels_filename,train_set_size)
    test_labels = extract_labels(test_labels_filename,test_set_size)

    validation_data, validation_labels, train_data, train_labels = splitData(train_data, train_labels)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def oneHot(result,maximum):
    out = []
    for i in range(maximum):
        out.append(0)
    out[result] = 1
    return out

def binaryCount(numberRecords=10000,rangeLength=10):
    """
    Small Dataset for LSTM and GRU
    """
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []

    random.seed(42)
    random.seed(42)
    for i in range(numberRecords):
        data = []
        for j in range(rangeLength):
            data.append(random.getrandbits(1))
        data=numpy.reshape(data,(rangeLength,1))
        label=oneHot(len(list(filter(lambda x: x==1,data))),rangeLength+1)
        
        if i%5==0:
            validation_data.append(data)
            validation_labels.append(label)
        else: 
            train_data.append(data)
            train_labels.append(label)
            
    return numpy.array(train_data), numpy.array(train_labels), numpy.array(validation_data), numpy.array(validation_labels), numpy.array(validation_data), numpy.array(validation_labels)