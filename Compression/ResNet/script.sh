
for i in 0 3 6 9 12 15 18 21 24 27 30 33 36 39 41 44 47 50 
do 
		echo "crf_value $i"
		python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ~/ImageNet/val2012/ --compressAtLayer 1 --compressAtBlock 1 --batch-size 32 --pretrained --CRFValue $i	

done
