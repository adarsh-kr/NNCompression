from shutil import copyfile
import numpy as np
#from numpy import genfromtxt
import wrap

def createFilteredData(filterFile, datadir="", className=""):
    with open(filterFile) as f:
        lines = f.readlines()
        for line in lines:
            img = str(int(line.strip()) + 1)
            img = "out" + img.zfill(3) + ".jpg"
            imgPath = datadir + "1/" + img
            copyfile(imgPath, datadir+ "/" + className + "/" + img)

def EncodeDecode(layerDump, height, width):

    data = np.genfromtxt(layerDump, delimiter=',')
    b, cols = data.shape
    assert cols==height*width
    
    wrap.compress(data, data.min(), data.max(), b, height, width, "random")
        
def calcAccuracy(gt, pl):

    with open(gt) as fl, open(pl) as pl:
        a1 = fl.readlines()
        a2 = pl.readlines()
       
        a1 = [float(x.strip()) for x in a1]
        a2 = [float(x.strip()) for x in a2]

        
        result = [x==y for x,y in zip(a1,a2)]
        return sum(result)/len(result)

if __name__ == "__main__":
    #createFilteredData("data/TopK/CaliforniaI_600/warplane_index", "data/CaliforniaI_600/", "warplane")
    
    ##to encode data, basically to get the video 
   # EncodeDecode("../../ResNet/layerDump/TestData/sample_100", 56*8, 56*8)
    #EncodeDecode("sample_100", 56*8, 56*8)
    for crf in [0, 3, 6, 9, 12 , 15, 18, 20, 23, 26, 29, 32]:
       print(crf, calcAccuracy("layerDump/TestData/labels-1_-1_0", "layerDump/TestData/labels1_1_{}".format(crf)))





