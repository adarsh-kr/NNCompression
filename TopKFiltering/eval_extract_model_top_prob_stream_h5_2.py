from __future__ import print_function
import os
import math
import argparse
import numpy as np
import cntk
import _cntk_py
import timeit
import tables

import cntk.io.transforms as xforms
from cntk.logging import *
from cntk.ops import *
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
from cntk.debugging import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_file_path', default="../datasets/Lausanne/models/ResNet18_R6_112_Stream.model", 
                    type=str, help = 'model file path')

parser.add_argument('--map_file_path', default="../datasets/Lausanne/mapping_file", 
                    type=str, help = 'map file path')

parser.add_argument('--num_objects', default=600, 
                    type=int, help = 'total images in the test dataset')

parser.add_argument('--image_size', default=112, 
                    type=int, help = 'image size')

parser.add_argument('--feature_size', default=112, 
                    type=int, help = 'feature size')

parser.add_argument('--num_classes', default=5, 
                    type=int, help = 'num classes')

parser.add_argument('--video_name', default='Lausanne',
                    type=str, help='name of the video stream')

    
args = parser.parse_args()
model_file_path = args.model_file_path
map_file_path   = args.map_file_path
feat_node_name  = ''
pred_node_name  = 'prediction'
num_objects     = args.num_objects
image_size   = args.image_size
feature_size = args.feature_size
mean_file_path = ''
num_classes    = args.num_classes
out_top_file_name  = os.path.join("../results/TopKFiltering/", args.video_name, "topKProb")
out_feat_file_name =  os.path.join("../results/TopKFiltering/", args.video_name, "featureVec")

# model_file_path = sys.argv[1]
# map_file_path = sys.argv[2]
# feat_node_name = sys.argv[3]
# pred_node_name = sys.argv[4]
# num_objects = int(sys.argv[5])
# image_size = int(sys.argv[6])
# feature_size = int(sys.argv[7])
# mean_file_path = sys.argv[8]
# num_classes = int(sys.argv[9])
# out_top_file_name = sys.argv[10]
# out_feat_file_name = sys.argv[11]

#top_k = min(num_classes, 100)
top_k = num_classes


def create_mb_source(image_height, image_width, num_channels, map_file, mean_file, is_training):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." % (map_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.875, jitter_type='uniratio') # train uses jitter
        ]
    else: 
        transforms += [
            xforms.crop(crop_type='center', side_ratio=0.875) # test has no jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),        
    ]

    if mean_file != '':
        transforms += [
            xforms.mean(mean_file),
        ]        

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features = StreamDef(field='image', transforms=transforms) # first column in map file is referred to as 'image'
            )),  
        randomize = is_training, 
        multithreaded_deserializer = True,
        max_sweeps = 1)

def eval_and_write(model_file_path, feat_node_name, pred_node_name, minibatch_source, 
                   num_objects, feature_size, out_top_file_name, out_feat_file_name):

    # Load the full model
    loaded_model = load_model(model_file_path)
    if (feat_node_name == ''):
        fl = loaded_model.find_all_with_name('')
        for f in fl:        
            if type(f) is cntk.ops.functions.Function:                
                if (f.outputs[0].shape == (feature_size, 1, 1)):
                    feat_node = f
                    print("Inside")
    else:
        feat_node = loaded_model.find_by_name(feat_node_name)

    feat_output_nodes  = combine([feat_node.owner])

    features = cntk.input_variable((feature_size, 1, 1))

    if (pred_node_name == ''):
        pred_node = loaded_model.outputs[0]
        pred_output_node  = combine([pred_node.owner])
        all_z = pred_output_node
        fl = pred_output_node.find_all_with_name('')
        for f in fl:        
            if type(f) is cntk.ops.functions.Function:
                if (f.inputs[0].shape == (feature_size, 1, 1, num_classes)):
                    z = f(features)
                    
    else:
        f = loaded_model.find_by_name(pred_node_name)                
        fc = f.clone(cntk.CloneMethod.freeze, {feat_node: placeholder(name='features')})
        z = fc(features)

    sm_out = softmax(z)

    top_out_f = tables.open_file(out_top_file_name, mode='w')
    feat_out_f = tables.open_file(out_feat_file_name, mode='w')
    atom = tables.Float64Atom()

    top_array_c = top_out_f.create_earray(top_out_f.root, 'data', atom, (0, 2 * top_k), expectedrows = num_objects)
    feat_array_c = feat_out_f.create_earray(feat_out_f.root, 'data', atom, (0, feature_size), expectedrows = num_objects)

    features_si = minibatch_source['features']    
    mb_size = 50

    total_elapsed = 0.0
    num_batch = int(num_objects/mb_size)
    if (0 != (num_objects % mb_size )):
        num_batch += 1
    img_idx = 0

    for i in range(0, num_batch):

        start_time = timeit.default_timer()

        cur_mb_size = mb_size
        if ((i == num_batch - 1) and (0 != (num_objects % mb_size ))):
            cur_mb_size = num_objects % mb_size
        mb = minibatch_source.next_minibatch(cur_mb_size)

        feat_output = feat_output_nodes.eval(mb[features_si])
        elapsed = timeit.default_timer() - start_time
        total_elapsed += elapsed

        feat_input = np.empty((cur_mb_size, feature_size, 1, 1), dtype = np.float32)
        for s in range(cur_mb_size):
            output_np = np.squeeze(feat_output[s])
            feat_array_c.append(output_np[None])
            feat_input[s] = output_np[...,np.newaxis,np.newaxis]

        start_time = timeit.default_timer()

        output = sm_out.eval({z.arguments[0]: feat_input})

        elapsed = timeit.default_timer() - start_time
        total_elapsed += elapsed

        del feat_output
        del feat_input

        for s in range(cur_mb_size):
            output_np = np.squeeze(output[s])
            sortinds = np.argsort(output_np)
            top_k_np = np.zeros(2 * top_k)
            for k in range(1, top_k + 1):
                top_k_np[2 * (k - 1)] = sortinds[-k]
                top_k_np[2 * (k - 1) + 1] = output_np[sortinds[-k]]
            top_array_c.append(top_k_np[None])
            del top_k_np
            img_idx += 1
            if (img_idx % 10000 == 0):                
                feat_out_f.flush()
                top_out_f.flush()
                print('Processed {} items'.format(img_idx))
                sys.stdout.flush()
        
        del output
        del mb
    
    print ("Execution time: {0}".format(total_elapsed))
    feat_out_f.close()
    top_out_f.close()    

if __name__ == '__main__':
    # define location of model and data and check existence
    base_folder = os.path.dirname(os.path.abspath(__file__))

    if not (os.path.exists(model_file_path)):
        print("The model file does notexist.")
        exit(0)

    # create minibatch source
    image_height = image_size
    image_width  = image_size
    num_channels = 3
    minibatch_source = create_mb_source(image_height, image_width, num_channels, \
                                        map_file_path, mean_file_path, False)
    
    eval_and_write(model_file_path, feat_node_name, pred_node_name, minibatch_source, 
                   num_objects, feature_size, out_top_file_name, out_feat_file_name)
