
for i in 3
do 
	for j in `seq 1 23`; 
	do
		echo "Starting new layer $i"
#	python inference.py --evalData /data/adarsh/layerDump/TrainData/LayerOutput_block_$i --evalLables /data/adarsh/layerDump/TrainData/labels --checkpoint checkpoint/checkpoint_best_acc_layer_$i.pth.tar --dumpData --dumpPath /data/adarsh/layerDump/TrainData/ --layerNum $i
		python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Lausanne/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 32 --pretrained
	done
done
