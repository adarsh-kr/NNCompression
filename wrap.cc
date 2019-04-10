#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include "FeatureMap.h"
#include "Codecs.h"

using namespace std;
namespace py = pybind11;

py::array_t<double> compress(py::array_t<double> data, double q_min, double q_max, int batch, int height, int width, string fileName, string preset_parameter="fast")
{
    py::buffer_info info = data.request();
    double *dataPtr = (double*) info.ptr;
    //data is one dimension of size B*H*W
    vector<FeatureMap> allFrames;
    bool first = true;

    for(int i=0; i<batch; i++)
    {   
        vector<double> frame;
        for(int j=0;j<height*width; j++)
            {
                frame.push_back(dataPtr[i*(height*width) + j]);
                // cout<<dataPtr[i*(height*width) + j]<<" ";
            }
        allFrames.push_back(FeatureMap(frame, height, width, q_min, q_max));
    }

    
    Codecs codecData(allFrames, height, width, q_min, q_max, batch, fileName, preset_parameter);
    codecData.EncodeVideo();
    vector<FeatureMap> out = codecData.DecodeVideo();
    
    py::array_t<double> finalOutput = py::array_t<double>(batch*height*width);
    py::buffer_info outBuf =  finalOutput.request();
    double *outDataPtr = (double *) outBuf.ptr;

    for(int i=0; i<out.size(); i++)
    {   
        // convert yuv to grayscale
        out[i].ConvertYUV2GrayScale();
        // convert grayscale to tensor
        out[i].ConvertGrayScale2Tensor();

        // finalOutput.insert(finalOutput.end(), out[i].tensor.begin(), out[i].tensor.end());
        for(int j=0; j<out[i].tensor.size(); j++)
            {
                outDataPtr[i*height*width + j] = out[i].tensor[j];     
            }
    }
    
    return finalOutput; 
}

PYBIND11_PLUGIN(wrap) {
    pybind11::module m("wrap", "auto-compiled c++ extension");
    m.def("compress", &compress);
    return m.ptr();
}
