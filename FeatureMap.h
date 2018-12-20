#ifndef FEATUREMAP_H
#define FEATUREMAP_H

#include <iostream>
#include <vector>
#include <tuple>
using namespace std;

struct GrayScaleFormat
{
    vector<int> grayscale;
};
struct RGBFormat
{
    vector<int> r, g, b;
};

struct YUVFormat
{
    vector<int> y, u, v;
};

class FeatureMap
{
    private:
       
        // get YUV for 
        tuple<double, double, double> GetYUV(int grayScale);
        // for yuv_420, we need to reduce resolution for u and v
        vector<int> ReduceResolution(const vector<int>& x);         
    
    public:
     // interlayer output/feature map
        vector<double> tensor;
        // height and width of images
        int H,W;
        // for quantization
        double q_min, q_max;
        // factor for scaling
        int scaleFactor = 255;
        // status of operations
        bool empty, rgb_done, yuv_done, grayscale_done;
        // store rgb format
        struct RGBFormat rgb_rep;
        // store YUV format
        struct YUVFormat yuv_rep, yuv_420;
        // store gray scale format
        struct GrayScaleFormat grayscale_rep;

        FeatureMap();
        FeatureMap(vector<double> data, int height, int width, double q_min, double q_max);
        FeatureMap(vector<int> Y, int height, int width, double min_val, double max_val);
        
        // YUV to grayscale after decoding is done
        void ConvertYUV2GrayScale();
        
        // GrayScle to Tensor, after decoding
        void ConvertGrayScale2Tensor();
        
        // conver the double featuremap to grayscale
        // will incur loss
        void ConvertTensor2GrayScale();
        
        // convert to RGB format
        void ConvertRGB();
        
        // convert to YUV format and then YUV420, chroma subsampling
        void ConvertYUV();
        
        // print representation
        void PrintRep(string format="yuv_420"); 
        
        // just print size of each format
        void GetSizes();
};

#endif