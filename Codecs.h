#ifndef CODECS_H
#define CODECS_H

#include <iostream>
#include "FeatureMap.h"
#include <vector>
extern "C"
{
    #include "libavutil/frame.h"
    #include "libavutil/imgutils.h"
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavformat/avformat.h>
    // #include <libswscale/swscale.h>
}

struct FrameResponse
{
    vector<FeatureMap> frames;
    int response;
};

class Codecs
{
    public:
        vector<FeatureMap> frames;
        string fileName, preset_parameter; 
        int height, width, batch;
        double q_min, q_max;
        //to add codec Id
        Codecs();
        Codecs(vector<FeatureMap> data, int height, int width, double min_val, double max_val, int batch, string file, string preset_parameter);

        // encode one frame, being called from Encode
        void EncodeFrame(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt, FILE *outfile);

        // encode the video
        void EncodeVideo();

        // decode the video
        // make it return vector<FeatureMap> 
        vector<FeatureMap> DecodeVideo();

        // decode the frame 
        // vector<FeatureMap> DecodeFrame(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame);
        FrameResponse DecodeFrame(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame);

        vector<FeatureMap> RunCodec();
};


#endif