from shutil import copyfile


def createFilteredData(filterFile, datadir="", className=""):
    with open(filterFile) as f:
        lines = f.readlines()
        for line in lines:
            img = str(int(line.strip()) + 1)
            img = "out" + img.zfill(3) + ".jpg"
            imgPath = datadir + "1/" + img
            copyfile(imgPath, datadir+ "/" + className + "/" + img)


if __name__ == "__main__":
    createFilteredData("data/TopK/CaliforniaI_600/warplane_index", "data/CaliforniaI_600/", "warplane")








