for layer in 0 10 20 30 40 50 60 70
do
	for crf_value in 0 30 40 
	do
	python3 detect.py --images ../../datasets/Bellevue/frames/0/ --det det --cmprsLayerNum $layer --bs 32 --topK 2 --crf_value $crf_value --videoName Bellevue --outputFile ../../results/YOLO/CompressionStats/ --runTillBatch 3
done
done
