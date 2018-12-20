#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include "FeatureMap.h"
#include "Codecs.h"

using namespace std;
namespace py = pybind11;

py::array_t<double> compress(py::array_t<double> data, double q_min, double q_max, int batch, int height, int width, string fileName)
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

    cout<<"Q_min"<<q_min<<endl;
    cout<<"Q_Max"<<q_max<<endl;
    cout<<allFrames.size()<<endl;
    
    Codecs codecData(allFrames, height, width, batch, fileName);
    codecData.EncodeVideo();
    codecData.DecodeVideo();
    return data; 
}

PYBIND11_PLUGIN(wrap) {
    pybind11::module m("wrap", "auto-compiled c++ extension");
    m.def("compress", &compress);
    return m.ptr();
}
