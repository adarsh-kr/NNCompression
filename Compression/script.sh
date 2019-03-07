
for i in 1 2 3 4
do 
	for j in `seq 0 2`; 
	do
		echo "Starting new layer $i"
#	python inference.py --evalData /data/adarsh/layerDump/TrainData/LayerOutput_block_$i --evalLables /data/adarsh/layerDump/TrainData/labels --checkpoint checkpoint/checkpoint_best_acc_layer_$i.pth.tar --dumpData --dumpPath /data/adarsh/layerDump/TrainData/ --layerNum $i
		#python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Lausanne/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 64 --pretrained
	        python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../results/TopKFiltering/Coral/filteredFrames_classId_0/ --compressAtLayer $i --compressAtBlock $j --batch-size 32 --pretrained
	done
done


for i in 3
do
        for j in `seq 0 23`;
        do
                echo "Starting new layer $i"
#       python inference.py --evalData /data/adarsh/layerDump/TrainData/LayerOutput_block_$i --evalLables /data/adarsh/layerDump/TrainData/labels --checkpoint checkpoint/checkpoint_best_acc_layer_$i.pth.tar --dumpData --dumpPath /data/adarsh/layerDump/TrainData/ --layerNum $i
                #python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Lausanne/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 64 --pretrained
                python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../results/TopKFiltering/Coral/filteredFrames_classId_0/ --compressAtLayer $i --compressAtBlock $j --batch-size 32 --pretrained
        done
done

mv LayerData_layerNum_* layerDump/TestData/Coral_32_topk/


for i in 1 2 3 4
do
        for j in `seq 0 2`;
        do
                echo "Starting new layer $i"
#       python inference.py --evalData /data/adarsh/layerDump/TrainData/LayerOutput_block_$i --evalLables /data/adarsh/layerDump/TrainData/labels --checkpoint checkpoint/checkpoint_best_acc_layer_$i.pth.tar --dumpData --dumpPath /data/adarsh/layerDump/TrainData/ --layerNum $i
                #python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Lausanne/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 64 --pretrained
                python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Coral/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 32 --pretrained
        done
done


for i in 3
do
        for j in `seq 0 23`;
        do
                echo "Starting new layer $i"
#       python inference.py --evalData /data/adarsh/layerDump/TrainData/LayerOutput_block_$i --evalLables /data/adarsh/layerDump/TrainData/labels --checkpoint checkpoint/checkpoint_best_acc_layer_$i.pth.tar --dumpData --dumpPath /data/adarsh/layerDump/TrainData/ --layerNum $i
                #python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Lausanne/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 64 --pretrained
                python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Coral/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 32 --pretrained
        done
done



mv LayerData_layerNum_* layerDump/TestData/Coral_32_no_topk/



for i in 1 2 3 4
do
        for j in `seq 0 2`;
        do
                echo "Starting new layer $i"
#       python inference.py --evalData /data/adarsh/layerDump/TrainData/LayerOutput_block_$i --evalLables /data/adarsh/layerDump/TrainData/labels --checkpoint checkpoint/checkpoint_best_acc_layer_$i.pth.tar --dumpData --dumpPath /data/adarsh/layerDump/TrainData/ --layerNum $i
                #python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Lausanne/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 64 --pretrained
                python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Coral/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 1 --pretrained
        done
done


for i in 3
do
        for j in `seq 0 23`;
        do
                echo "Starting new layer $i"
#       python inference.py --evalData /data/adarsh/layerDump/TrainData/LayerOutput_block_$i --evalLables /data/adarsh/layerDump/TrainData/labels --checkpoint checkpoint/checkpoint_best_acc_layer_$i.pth.tar --dumpData --dumpPath /data/adarsh/layerDump/TrainData/ --layerNum $i
                #python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Lausanne/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 64 --pretrained
                python3 main.py --evalMode --arch resnet101 --evalOnVideoData --video_name ../datasets/Coral/frames/ --compressAtLayer $i --compressAtBlock $j --batch-size 1 --pretrained
        done
done

