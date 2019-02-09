import cv2
import h5py
import numpy as np
import argparse, shutil, os 

def getBackgroundImage(folder):
    
    print('starting to get the background img')
    if os.path.exists(os.path.join(folder, "../avg_frame.jpg")):
        print("Backgound Image already exists")
        return    
    
    first = True
    count = 0
    for filename in os.listdir(folder):
        if first:
            first = False
            avg_frame = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_UNCHANGED)  
            backgroundImg = avg_frame.astype(np.int)

        else :
            backgroundImg +=avg_frame.astype(np.int)
            
        count+=1
    
    backgroundImg = backgroundImg/count
    cv2.imshow("img", backgroundImg.astype(np.uint8))
    cv2.waitKey(5000)
    cv2.imwrite(os.path.join(folder, "../avg_frame.jpg"), backgroundImg)
    print('background img done')
    print(count)
     
# creates a text file from h5 file 
def create_topK_file(file, outfile):
    f = h5py.File(file,'r')
    data = list(f['data'])
    
    with open(outfile, 'w') as writer:
        for i in range(len(data)):
            np.savetxt(writer, data[i].reshape(1,-1))

# takes topKtxt file, value of k, and class index
# creates a file containing index of images which are in topK
def topKFilter(txtFile, k, classId, outputFile, obj_map):
    data = np.loadtxt(txtFile, delimiter=' ')
    print(data.shape)
    print(outputFile)
    filteredFrameids = []
    with open(outputFile, "w") as writer, open(obj_map) as f:
        for i in range(data.shape[0]):
            frame_id = f.readline().strip().split("\t")[1]
            line = data[i,].tolist()
            # indexes
            classIdIndex = line[::2].index(classId)
            if classIdIndex+1 <= k:
                # print(classIdIndex+1, k)
                filteredFrameids = filteredFrameids + [frame_id]
                # writer.write(str(i+1) + "\n")
            else:
                print(classIdIndex+1, k)

        filteredFrameids = list(set(filteredFrameids))
        for x in filteredFrameids:
            writer.write(str(x) + "\n")

# just copy the filtered files to outputfolder
def moveFilteredFiles(filteredFile, frameFolder, outputFolder):
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        print("Output Folder created !! ")

    with open(filteredFile) as f:
        data = f.readlines()
        # traverse each image path
        count = 0
        for x in data:
            imgname = "out" + "{:04d}".format(int(x)) + ".jpg"
            path = frameFolder + "/" + imgname
            outputPath = outputFolder + "/" + imgname
            shutil.copyfile(path, outputPath)
            count +=1 
        print("{} Frames Filtered".format(count))
        print("Filtered File Move Done")

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="TopK Filtering")
    parser.add_argument('--k', default=1, type=int, help = 'value for K, in top K')
    parser.add_argument('--classId', default=0, type = int, help="index of the class")
    parser.add_argument('--video_name', default="Lausanne", type = str, help="video name")

    args = parser.parse_args()
    
    create_topK_file("../results/TopKFiltering/{}/topKClassProb".format(args.video_name), "../results/TopKFiltering/{}/topKClassProb.txt".format(args.video_name))
    
    topKFilter("../results/TopKFiltering/{}/topKClassProb.txt".format(args.video_name)
                ,args.k
                ,args.classId
                ,"../results/TopKFiltering/{0}/filteredFramesIds_classId_{1}.txt".format(args.video_name, args.classId)
                ,"../results/TopKFiltering/{0}/obj_map.txt".format(args.video_name))
    
    moveFilteredFiles("../results/TopKFiltering/{0}/filteredFramesIds_classId_{1}.txt".format(args.video_name, args.classId),
                      "/home/adarsh/Desktop/RAWork/NNCompression/datasets/{}/frames/0/".format(args.video_name),
                      "/home/adarsh/Desktop/RAWork/NNCompression/results/TopKFiltering/{0}/filteredFrames_classId_{1}".format(args.video_name, args.classId))


    # getBackgroundImage("../datasets/Lausanne/frames/")

