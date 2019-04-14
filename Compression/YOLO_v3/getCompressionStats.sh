declare -a arr=("Lausanne" "Auburn" "Bellevue" "Coral")
declare -a obj=("car" "person" "truck")

## now loop through the above array
#for i in "${arr[@]}"
#do
#	for j in "${obj[@]}"
#	do
#	   	 echo "$i"
#	      	 # or do whatever with individual element of the array
#	         
#		 if [ ! -d ../../results/YOLO/CompressionStats/$i ] 
#		 then
#		 	mkdir ../../results/YOLO/CompressionStats/$i
#		 fi
#
#		 if [ -d ../../results/YOLO/TopKTinyYolo/$i/filteredFrames/$j ]
#		 then
#			if [ ! -d ../../results/YOLO/CompressionStats/$i/$j ]
#			then 
#				mkdir ../../results/YOLO/CompressionStats/$i/$j
#			fi	
#		 	python3 detect.py --det det --images ../../results/YOLO/TopKTinyYolo/$i/filteredFrames/$j --videoName $i --outputFile /users/adarsh/NNCompression/results/YOLO/CompressionStats/ --bs 32 
#			mv LayerOutput_* > /users/adarsh/NNCompression/results/YOLO/CompressionStats/$i/$j
#		 fi
#		 
#

#	done
#done


for crf_value in 0 10 20 30 50:
do
	python3 detect.py --det det --images ../../results/YOLO/TopKTinyYolo/Bellevue/filteredFrames/person --videoName Bellevue --outputFile ../../results/YOLO/CompressionStats/ --bs 32 --crf_value $crf_value --cmprsLayerNum -1 
done
