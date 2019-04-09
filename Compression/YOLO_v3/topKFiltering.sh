declare -a arr=("Lausanne" "Coral" "Auburn" "Bellevue")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   # or do whatever with individual element of the array
   python3 detect.py  --det det --images ../../datasets/$i/frames/0/ --videoName $i --weights yolov3-tiny.weights --cfg cfg/yolov3-tiny.cfg --outputFile /users/adarsh/NNCompression/results/YOLO/TopKTinyYolo/ --topK 4 
done

#python3 detect.py  --det det --images ../../datasets/Lausanne/frames/0/ --videoName Lausanne
python3 utils.py --videoName Lausanne --objClass person --TopKFilter 
python3 utils.py --videoName Auburn   --objClass person --TopKFilter
python3 utils.py --videoName Auburn   --objClass car --TopKFilter
python3 utils.py --videoName Auburn   --objClass truck --TopKFilter
python3 utils.py --videoName Bellevue --objClass person --TopKFilter
python3 utils.py --videoName Bellevue --objClass car --TopKFilter
python3 utils.py --videoName Bellevue --objClass truck --TopKFilter
python3 utils.py --videoName Coral --objClass person --TopKFilter




