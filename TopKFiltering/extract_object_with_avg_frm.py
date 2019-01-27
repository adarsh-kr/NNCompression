import sys
# import imutils
import time
import cv2
import numpy as np
import os
import subprocess
import argparse


parser = argparse.ArgumentParser(description="Get Object Images")
parser.add_argument('--area_limit', default=2000, type=int, help = 'min area of the bounding box')
parser.add_argument('--frm_path', default="../datasets/Lausanne/frames/", type = str, help="from path")
parser.add_argument('--output_path', default="../results/TopKFiltering/Lausanne/objectImgs/", type = str, help="output folder")

# area_limit = int(sys.argv[1])
# frm_path = sys.argv[2]
# output_path = sys.argv[3]


if __name__ == "__main__":
    args = parser.parse_args()
    
    cv2.ocl.setUseOpenCL(False)
    avg_frame = cv2.imread(os.path.join(args.frm_path, '..',  'avg_frame.jpg'), cv2.IMREAD_UNCHANGED)
    avg_frame = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)    
    
    if not (os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    
    obj_map_file = open(os.path.join(args.output_path, "..", 'obj_map.txt'), 'w')
    mapping_file = open(os.path.join(args.output_path, "..", 'mapping_file.txt'), 'w')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dil = np.ones((20,20), np.uint8)

    # Go through the frame map file
    for line in open(os.path.join(args.frm_path, '..', 'frm_map.txt')):
        print(line)
        tokens = line.rstrip().split()
        frame_idx = int(tokens[0])
        image = cv2.imread(os.path.join(args.frm_path, tokens[1]))
        # filter_image = cv2.bilateralFilter(image, 7, 150, 150)
        filter_image = image
        gray_image   = cv2.cvtColor(filter_image, cv2.COLOR_BGR2GRAY)
        #gray_image  = cv2.GaussianBlur(gray_image, (9, 9), 0)

        fgmask = cv2.absdiff(gray_image, avg_frame)
        ret, fgmask = cv2.threshold(fgmask,  25, 255, cv2.THRESH_BINARY)
        #fgmask[fgmask < 240] = 0
        cv2.imwrite(os.path.join(args.output_path, 'tmp.jpg'), fgmask)

        # Fill any small holes
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        #dilation = cv2.dilate(opening, kernel, iterations=2)
        dilation = cv2.dilate(opening, kernel_dil, iterations=1)

        #_, thresholded = cv2.threshold(dilation, 0, 255, 
        #                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        
        th1 = dilation

        #ret,th1 = cv2.threshold(fgmask,25,255,cv2.THRESH_BINARY)
        #th1 = cv2.dilate(th1, None, iterations=4)


        #th1 = fgmask
        #kernel = np.ones((11,11),np.uint8)
        #close_operated_image = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
        #cv2.imwrite(os.path.join(output_path, 'tmp1.jpg'), close_operated_image)
        #_, thresholded = cv2.threshold(close_operated_image, 0, 255, 
        #                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        median = cv2.medianBlur(th1, 5)
        th1 = median
        cv2.imwrite(os.path.join(args.output_path, 'tmp2.jpg'), th1)
        # _,contours,hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1) #CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1) #CHAIN_APPROX_SIMPLE)

        obj_idx = 0
        
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if (w * h >= args.area_limit):
                obj_img_path = 'frame_{0:07d}_obj_{1:04d}.jpg'.format(frame_idx, obj_idx)
                abs_obj_path = os.path.abspath(os.path.join(
                    args.output_path, obj_img_path))
                cv2.imwrite(abs_obj_path, image[y:y+h, x:x+w])
                map_str = '{}\t{}\t{},{},{},{}\n'.format(
                    abs_obj_path, frame_idx, x, x+w, y, y+h)
                obj_map_file.write(map_str)
                obj_idx += 1

        del image
        del gray_image
        del fgmask  
        # break

    obj_map_file.close()
