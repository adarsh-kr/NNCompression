#include "FeatureMap.h"
#include <iostream>
#include <vector>
using namespace std;

tuple<double, double, double> FeatureMap::GetYUV(int grayScale)
{
    // all r,g,b are same as grayscale
    double y =  (0.299 + 0.587 + 0.114)*grayScale;
    double u =  (-0.147- 0.289 + 0.436)*grayScale;
    double v =  (0.615 - 0.515 - 0.100)*grayScale;
    return make_tuple(y, u, v);
}

vector<int> FeatureMap::ReduceResolution(const vector<int>& x)
{
    // allocate that much size 

    int final_size = H*W/4;
    vector<int> out;

    for(int i=0; i<H-1; i=i+2) // incrementing by 2 here
    {
        for(int j=0 ; j<W-1; j=j+2)
            {
                // take average of four points
                int index_1 = i*W + j;
                int index_2 = index_1 + 1;
                int index_3 = (i+1)*W + j;
                int index_4 = index_3 + 1;
                // take average of all these values
                double avg = (x[index_1] + x[index_2] + x[index_3] + x[index_4])/4;
                out.push_back(avg); 
                // cout<<i<<" "<<j<<"Avg: "<<avg<<endl;
            }
    }
    return out;
}

FeatureMap::FeatureMap()
{
    empty = true;
    rgb_done = false;
    grayscale_done = false;
    yuv_done = false;
}

FeatureMap::FeatureMap(vector<double> data, int height, int width, double min_val, double max_val)
{
    H=height;
    W=width;
    tensor = data;
    // boolean to check whether the tensor has data or not 
    empty = true;
    q_min = min_val;
    q_max = max_val;
    rgb_done = false;
    yuv_done = false;
    grayscale_done=false;
}

// only handing the case for GrayScale
FeatureMap::FeatureMap(vector<int> Y, int height, int width, double min_val, double max_val)
{
    H = height;
    W = width;
    yuv_420.y = Y;
    yuv_rep.y = Y;

    // set YUV420 u and v to 0
    yuv_420.u = vector<int>(height*width/4, 0);
    yuv_420.v = vector<int>(height*width/4, 0);

    // set YUV u and v to 0
    yuv_rep.u = vector<int>(height*width, 0);
    yuv_rep.v = vector<int>(height*width, 0);

    q_max = max_val;
    q_min = min_val;
}

// convert tensor 2 grayscale
void FeatureMap::ConvertTensor2GrayScale()
{
    for(auto i=tensor.begin(); i!=tensor.end(); i++)
        {
            // here double is going to get coverted into int
            grayscale_rep.grayscale.push_back((scaleFactor*(*i-q_min))/(q_max-q_min)); 
        }
    
    grayscale_done = true;
}

//recover the grayscale from the YUV  
void FeatureMap::ConvertYUV2GrayScale()
{
    // just handle Y, as grayscale
    grayscale_rep.grayscale = yuv_rep.y;
    grayscale_done = true;
}

// get the tensor back after decoding
void FeatureMap::ConvertGrayScale2Tensor()
{
    for(int i=0; i<grayscale_rep.grayscale.size(); i++)
        {
            
            double buf = (grayscale_rep.grayscale[i]*(q_max - q_min))/255 + q_min;
            tensor.push_back(buf);
        }    
}

void FeatureMap::ConvertYUV()
{
    if(!grayscale_done)
            ConvertTensor2GrayScale();

    for(int i=0; i<grayscale_rep.grayscale.size(); i++)
        {
            tuple<double, double, double> x = GetYUV(grayscale_rep.grayscale[i]);
            // y is going to be same as grayscale
            yuv_rep.y.push_back(get<0>(x));
            // u and v are going to be zero btw
            yuv_rep.u.push_back(get<1>(x));
            yuv_rep.v.push_back(get<2>(x));
        }

    // y with full resolution
    yuv_420.y = yuv_rep.y;
    // reduce resolution of U and V
    yuv_420.u = ReduceResolution(yuv_rep.u);
    yuv_420.v = ReduceResolution(yuv_rep.v);
    
    yuv_done = true;
}
