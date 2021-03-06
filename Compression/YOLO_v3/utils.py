import os, argparse
import os.path
import numpy as np
from shutil import copyfile
import subprocess
import time
import re
from shutil import copyfile

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
            


def YOLOCompressionPlot(videoName, objClass, dir="../../results/YOLO/CompressionStats", fileFormat="LayerOutputConv_{0}", num_layers=105, crf_value="0"):

    layerAvgCompression = []
    layerAvgTime = []
    layerSumCompression = []
    
    #dir = os.path.join(dir, videoName, objClass)
    dir = os.path.join(dir, videoName, "crf_value_{}_preset_value_ultrafast".format(crf_value))
    

    for i in range(num_layers):
        if i==82 or i==94:
            continue
        file = dir + "/" + fileFormat.format(i)
        with open(file) as f:
            data = np.loadtxt(file, delimiter=",")
            avgCompression = data.mean(axis=0)[0]
            avgTime = data.mean(axis=0)[1]
            totalCompression = data.sum(axis=0)[0]
            layerAvgCompression = layerAvgCompression + [avgCompression]
            layerAvgTime = layerAvgTime + [avgTime]
            layerSumCompression = layerSumCompression + [totalCompression] 

    print(layerAvgCompression)
    print(layerAvgTime)
    print(layerSumCompression)
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

def TopkFilteredFiles(videoName, objClass, resultDir="../../results/YOLO/TopKFiltering", frameDir="../../datasets/"):
   resultFile = os.path.join(resultDir, videoName, "result_name.txt")
   frameDir = os.path.join(frameDir, videoName, "frames/0/")
   outDir = os.path.join(resultDir, videoName, "filteredFrames", objClass)
   
   if not os.path.isdir(outDir):
       os.mkdir(outDir)

   count = 0
   
   with open(resultFile) as f:
       lines = f.readlines()
       for line in lines:
           parsedLine = re.split('_|,| ', line.strip())
           imgName = parsedLine[0]
           
           if objClass in parsedLine:
               src = os.path.join(frameDir, imgName)
               dst = os.path.join(outDir, "out{:04d}.jpg".format(count))
               copyfile(src, dst)
               count += 1
           
           if count%100==0:
               print("Count Done {}".format(count))


def CalcRecall(videoName, objClass, resultDir="../../results/YOLO/TopKTinyYolo", realLabels="../../results/YOLO/RealLabels"):
    groundTruth = os.path.join(realLabels, videoName, "result_name.txt")
    miniYolo = os.path.join(resultDir, videoName, "result_name.txt")

    with open(groundTruth) as file1, open(miniYolo) as file2:
        totalCount = 0
        presentCount = 0
        
        groundDict = {}
        yoloDict = {}
        
        for line1 in file1:
            parsedLine1 = line1.strip().split(',')
            groundDict[parsedLine1[0]]=[x.split("_")[0] for x in parsedLine1]

        for line2 in file2:
            parsedLine2 = re.split('_|,| ', line2.strip())
            yoloDict[parsedLine2[0]] = parsedLine2
        
        for key, value in groundDict.items():

            if objClass in groundDict[key]:
                totalCount+=1
                if key in yoloDict.keys():
                    presentCount+=1


            
        print("Recall {0}, {1}, {2}".format(presentCount/totalCount, presentCount, totalCount))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--videoName",  help = "Image / Directory containing images to perform detection upon", default = "imgs", type = str)
    parser.add_argument("--TopKFilter", action='store_true')
    parser.add_argument("--resultDir",  type=str, default="../../results/YOLO/TopKTinyYolo/")
    parser.add_argument("--frameDir",   type=str, default="../../datasets/")
    parser.add_argument("--objClass",   type=str, default="person")
    parser.add_argument('--recall', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--crf_value', type=str)


    args = parser.parse_args()
    
    if args.TopKFilter:
        TopkFilteredFiles(args.videoName, args.objClass, resultDir=args.resultDir, frameDir=args.frameDir)
    if args.recall:
        CalcRecall(args.videoName, args.objClass)
    if args.plot:
        YOLOCompressionPlot(videoName=args.videoName,objClass=args.objClass, crf_value=args.crf_value)
    #ResNetAvgCompressionPlot()
    # BaselineVideoSize("Compression/data/CaliforniaI_600/1/", 1)    
    # MobileNetAvgCompressionPlot()
    # makeSequential("results/TopKFiltering/Bellevue/filteredFrames_classId_0/0")



