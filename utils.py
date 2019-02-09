import os
import numpy as np
from shutil import copyfile

# make sequential
def makeSequential(dir):

    outputDir = os.path.join(dir, "../sequential")

    count = 0
    for i in range(1200):
        src_file = os.path.join(dir, "out{:04d}.jpg".format(i))
        # print(src_file)
        if os.path.exists(src_file):
            print(count)
            dest_file = os.path.join(outputDir, "out{:04d}.jpg".format(count))
            count = count + 1
            copyfile(src_file, dest_file)
            


# plot resnet plot 
def ResNetAvgCompressionPlot(dir="results/Compression/California/batch_size_32"
                            , layers=[3, 4, 23, 3], fileFormat="LayerData_layerNum_{0}_blockNum_{1}"):
    # layers = [2, 2, 2, 2]
    # layers = [3, 4, 6, 3]
    # layers = [3, 4, 23, 3]
    # layers = [3, 8, 36, 3]

    layerAvgCompression = []
    layerAvgTime = []
    layerTotalCompression = []

    for i in range(len(layers)):
        for j in range(layers[i]):
            # file = os.path.join(dir, fileFormat.format(i+1, j))
            print(i,j)
            file = dir + "/" + fileFormat.format(i+1, j)
            with open(file) as f:
                data = np.loadtxt(file, delimiter=",")
                avgCompression = data.mean(axis=0)[0]
                avgTime = data.mean(axis=0)[1]
                layerAvgCompression = layerAvgCompression + [avgCompression]
                layerAvgTime = layerAvgTime + [avgTime]
                layerTotalCompression = layerTotalCompression + [data.sum(axis=0)[0]]

    print("Average Compression")
    print(layerAvgCompression)
    print("Total Compression")
    print(layerTotalCompression)
    print("Avg Time")
    print(layerAvgTime)

# plot resnet plot 
def MobileNetAvgCompressionPlot(dir="models/results/CaliforniaI_600/1fps/MobileNet/batchSize_10/"
                            , layers=17, fileFormat="Layer_{}"):

    layerAvgCompression = []
    layerAvgTime = []
    layerTotalCompression = []

    for j in range(layers):
        # file = os.path.join(dir, fileFormat.format(i+1, j))
        file = dir + "/" + fileFormat.format(j+1)
        with open(file) as f:
            data = np.loadtxt(file, delimiter=",")
            avgCompression = data.mean(axis=0)[0]
            avgTime = data.mean(axis=0)[1]
            layerAvgCompression = layerAvgCompression + [avgCompression]
            layerAvgTime = layerAvgTime + [avgTime]
            layerTotalCompression = layerTotalCompression + [data.sum(axis=0)[0]]
    
    print(layerAvgCompression)
    print(layerAvgTime)
    print(layerTotalCompression)



if __name__ == "__main__":
    ResNetAvgCompressionPlot()
    # MobileNetAvgCompressionPlot()
    # makeSequential("results/TopKFiltering/Bellevue/filteredFrames_classId_0/0")



