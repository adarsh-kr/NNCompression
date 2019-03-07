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
        


if __name__ == "__main__":
    #createFilteredData("data/TopK/CaliforniaI_600/warplane_index", "data/CaliforniaI_600/", "warplane")
    
    ##to encode data, basically to get the video 
   # EncodeDecode("../../ResNet/layerDump/TestData/sample_100", 56*8, 56*8)
    EncodeDecode("sample_100", 56*8, 56*8)






