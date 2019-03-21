import os
import os.path
import numpy as np
from shutil import copyfile
import subprocess
import time

def BaselineVideoSize(dir, batchsize):
    files_path = sorted([dir + x for x in os.listdir(dir)])
    count = 0
    totalData = 0
    frames = 0
    for img in files_path:
        # if os.path.exists(img):
        #     print("asa")
        frames = frames + 1
        copyfile(img, "delete/out{:03d}.jpg".format(count))
        count = count + 1
        if count >= batchsize:
            # a = subprocess.run([""])
            os.system("ffmpeg -i {0} -c:v libx264 -pix_fmt yuv420p delete/output.mp4".format("delete/out%03d.jpg"))
            count = 0
            totalData = totalData + os.path.getsize("delete/output.mp4")
            # delete all files
            print(frames)
            print(totalData)
            if os.path.exists('delete/output.mp4'):
                print("Removing")
                os.remove("delete/output.mp4")
            
            # time.sleep(2)

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
            


def YOLOCompressionPlot(dir=".", fileFormat="LayerOutput_{0}", num_layers=60):

    layerAvgCompression = []
    layerAvgTime = []
    layerSumCompression = []
    for i in range(num_layers):
        file = dir + "/" + fileFormat.format(i)
        with open(file) as f:
            data = np.loadtxt(file, delimiter=",")
            avgCompression = data.mean(axis=0)[0]
            avgTime = data.mean(axis=0)[1]
            layerAvgCompression = layerAvgCompression + [avgCompression]
            layerAvgTime = layerAvgTime + [avgTime]
    

    print(layerAvgCompression)
    print(layerAvgTime)
# plot resnet plot 
def ResNetAvgCompressionPlot(dir="Compression/"
                            , layers=[3, 3, 23, 3], fileFormat="LayerData_layerNum_{0}_blockNum_{1}"):
    # layers = [2, 2, 2, 2]
    # layers = [3, 4, 6, 3]
    # layers = [3, 4, 23, 3]
    # layers = [3, 8, 36, 3]

    layerAvgCompression = []
    layerAvgTime = []
    layerTotalCompression = []
    layerAvgRMSE = []
    layerAvgRMSE_II = []

    for i in range(len(layers)):
        for j in range(layers[i]):
            # file = os.path.join(dir, fileFormat.format(i+1, j))
            print(i,j)
            file = dir + "/" + fileFormat.format(i+1, j)
            with open(file) as f:
                data = np.loadtxt(file, delimiter=",")
                avgCompression = data.mean(axis=0)[0]
                avgTime = data.mean(axis=0)[1]
                avgMeanSqr = data.mean(axis=0)[2]
                avgMeanSqr_II = data.mean(axis=0)[3]
                layerAvgCompression = layerAvgCompression + [avgCompression]
                layerAvgTime = layerAvgTime + [avgTime]
                layerTotalCompression = layerTotalCompression + [data.sum(axis=0)[0]]
                layerAvgRMSE = layerAvgRMSE + [avgMeanSqr]
                layerAvgRMSE_II = layerAvgRMSE_II + [avgMeanSqr_II]

    print("Average Compression")
    print([x/1000000 for x in layerAvgCompression])
    print("Total Compression")
    print([x/1000000 for x in layerTotalCompression])
    print("Avg Time")
    print([x for x in layerAvgTime])
    print("Avg RMSE")
    print([x for x in layerAvgRMSE])
    print("Avg RMSE _II")
    print([x for x in layerAvgRMSE_II])

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
    YOLOCompressionPlot()
    #ResNetAvgCompressionPlot()
    # BaselineVideoSize("Compression/data/CaliforniaI_600/1/", 1)    
    # MobileNetAvgCompressionPlot()
    # makeSequential("results/TopKFiltering/Bellevue/filteredFrames_classId_0/0")



