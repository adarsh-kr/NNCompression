#declare -a arr=("Lausanne" "Coral" "Auburn" "Bellevue")

### now loop through the above array
#for i in "${arr[@]}"
#do
 #  echo "$i"
  # # or do whatever with individual element of the array
  # python3 detect.py  --det det --images ../../datasets/$i/frames/0/ --videoName $i

#done


## now loop through the above array
   # or do whatever with individual element of the array
python3 detect.py  --det det --images ../../datasets/COCO/ --videoName COCO




