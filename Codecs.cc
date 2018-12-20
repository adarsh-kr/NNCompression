#include <iostream>
#include "FeatureMap.h"
#include "Codecs.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <typeinfo>

extern "C"
{
    #include "libavutil/frame.h"
    #include "libavutil/imgutils.h"
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavformat/avformat.h>
    // #include <libswscale/swscale.h>
}

using namespace std;

void save_gray_frame(unsigned char *buf, int wrap, int xsize, int ysize, char *filename)
{
    FILE *f;
    int i;
    f = fopen(filename,"w");
    // writing the minimal required header for a pgm file format
    // portable graymap format -> https://en.wikipedia.org/wiki/Netpbm_format#PGM_example
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    // writing line by line
    for(i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}


struct FrameResponse
{
    vector<FeatureMap> frames;
    int response;
};


Codecs::Codecs()
{

}

Codecs::Codecs(vector<FeatureMap> data, int height, int width, double min_val, double max_val, int batch, string file)
{
    frames = data;
    fileName = file;
    this->height = height;
    this->width = width;
    this->batch = batch;
    this->q_min = min_val;
    this->q_max = max_val;
}


void Codecs::EncodeFrame(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
                   FILE *outfile)
{
    int ret;
    /* send the frame to the encoder */
    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        // fprintf(stderr, "error sending a frame for encoding\n");
        cerr<<"error sending a frame for encoding\n";
        exit(1);
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "error during encoding\n");
            exit(1);
        }
        
        printf("encoded frame %3"PRId64" (size=%5d)\n", pkt->pts, pkt->size);
        fwrite(pkt->data, 1, pkt->size, outfile);
        // stream>>pkt->data;
        av_packet_unref(pkt);
    }
}

void Codecs::EncodeVideo()
{      

    const AVCodec *codec;

    // // endcode for mpeg
    uint8_t endcode[] = { 0, 0, 1, 0xb7 };

    const char* filename = this->fileName.c_str();
    
    // register all codecs
    avcodec_register_all();
    
    /* find the mpeg1video encoder */
    codec = avcodec_find_encoder(AV_CODEC_ID_MPEG1VIDEO);
    
    if (!codec) {
        cerr<<"codec not found"<<endl;
        exit(1);
    }

    AVCodecContext *c = avcodec_alloc_context3(codec);
    AVFrame *picture  = av_frame_alloc();

    AVPacket *pkt     = av_packet_alloc();

    if (!pkt)
        exit(1);

    /* put sample parameters */
    c->bit_rate = 400000;
    /* resolution must be a multiple of two */
    c->width = this->width;
    c->height = this->height;
    /* frames per second */
    c->time_base = (AVRational){1, 25};
    c->framerate = (AVRational){25, 1};
    c->gop_size = 10; /* emit one intra frame every ten frames */
    c->max_b_frames = 1;
    // TODO : can it take other formats 
    c->pix_fmt = AV_PIX_FMT_YUV420P;



    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0) {
        cerr<<"could not open codec\n";
        exit(1);
    }    
    
    FILE *f;
    f = fopen(filename, "wb");

    if (!f) {
        cerr<<"could not open %s\n";
        exit(1);
    }
    
    picture->format = c->pix_fmt;
    picture->width  = c->width;
    picture->height = c->height;
    

    cout<<"!Encoder : Linsize"<<picture->linesize[0]<<" "<<picture->linesize[1]<<" "<<picture->linesize[2]<<" "<<picture->height<<" "<<picture->width<<endl;
    int ret = av_frame_get_buffer(picture, 32);
    if (ret < 0) {
        cerr<<"could not alloc the frame data\n";
        exit(1);
    }

    /* encode layers data*/
    for(int i=0; i<frames.size(); i++) {
        
        cout<<"Frame Num :"<<i<<endl;
        fflush(stdout);
        // compute the yuv_420 rep of each frame 
        frames[i].ConvertYUV();
        /* make sure the frame data is writable */
        ret = av_frame_make_writable(picture);

        if (ret < 0)
            exit(1);
        
        cout<<"Encoder : Linsize "<<picture->linesize[0]<<" "<<picture->linesize[1]<<" "<<picture->linesize[2]<<" "<<picture->height<<" "<<picture->width<<endl;
        /* prepare a dummy image */
        /* Y */
        for(int y=0; y<c->height; y++){
            for(int x=0; x<c->width; x++) {
                picture->data[0][y * picture->linesize[0] + x] = frames[i].yuv_420.y[ y*c->width+x];
            }
        }
        /* Cb and Cr */
        for(int y=0;y<c->height/2;y++) {
            for(int x=0;x<c->width/2;x++) {
                picture->data[1][y * picture->linesize[1] + x] = frames[i].yuv_420.u[y*c->width/2+x];
                picture->data[2][y * picture->linesize[2] + x] = frames[i].yuv_420.v[y*c->width/2+x];
            }
        }

        cout<<"Check point 2"<<endl;

        picture->pts = i;
        /* encode the image */
        EncodeFrame(c, picture, pkt, f);
    }
    
    // /* flush the encoder */
    EncodeFrame(c, NULL, pkt, f);
    /* add sequence end code to have a real MPEG file */
    fwrite(endcode, 1, sizeof(endcode), f);
    // f>>endcode;
    fclose(f);
    // f.close()
    avcodec_free_context(&c);
    av_frame_free(&picture);
    av_packet_free(&pkt);
}


vector<FeatureMap> Codecs::DecodeVideo()
{
    // register all codec
    avcodec_register_all();
    av_register_all();

    AVFormatContext* pFormatCtx = avformat_alloc_context();

    if (avformat_open_input(&pFormatCtx, this->fileName.c_str(), NULL, NULL)!=0)
    {
        cerr<<"Issue Opening FIle!"<<endl;
        exit(1);
    }

    cout<<"Format "<<pFormatCtx->iformat->name<<endl;
    cout<<"Duration "<<pFormatCtx->duration<<endl;
    cout<<"BitRate "<<pFormatCtx->bit_rate<<endl;

    if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
	{
		cerr<<"Error getting stream info!"<<endl;
        exit(1);
	}

    AVCodec *pCodec = NULL;
    AVCodecParameters *pCodecParameters =  NULL;
    int video_stream_index = -1;
    
    // get the codec and codec parameters for video stream 
    for(int i = 0; i < pFormatCtx->nb_streams; i++)
    {
        if(pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                cout<<"Video Stream Index "<<i<<endl;
                video_stream_index = i;
                pCodec = avcodec_find_decoder(pFormatCtx->streams[i]->codecpar->codec_id);
                pCodecParameters = pFormatCtx->streams[i]->codecpar;
            }
    }

    AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
    
    if (!pCodecContext)
    {
        cout<<"failed to allocated memory for AVCodecContext"<<endl;
        exit(1);
    }

    if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0)
    {
        cout<<"failed to copy codec params to codec context"<<endl;
        exit(1);
    }

    cout<<"Codec Pix Fmt :"<<pCodecContext->pix_fmt<<endl;

    if (avcodec_open2(pCodecContext, pCodec, NULL) < 0)
    {
        cout<<"failed to open codec through avcodec_open2"<<endl;
        exit(1);
    }

    AVFrame *pFrame = av_frame_alloc();
    if (!pFrame)
    {
        cout<<"failed to allocated memory for AVFrame"<<endl;
        exit(1);
    }

    AVPacket *pPacket = av_packet_alloc();
    if (!pPacket)
    {
        cout<<"failed to allocated memory for AVPacket"<<endl;
        exit(1);
    }

    int response = 0;

    // fill the Packet with data from the Stream
    // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga4fdb3084415a82e3810de6ee60e46a61
    while (av_read_frame(pFormatCtx, pPacket) >= 0)
    {
        // if it's the video stream
        if (pPacket->stream_index == video_stream_index) 
        {
            cout<<"AVPacket->pts "<< pPacket->pts;
            response = this->DecodeFrame(pPacket, pCodecContext, pFrame);
            
            if (response < 0)
                break;
        }
        // https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html#ga63d5a489b419bd5d45cfd09091cbcbc2
        av_packet_unref(pPacket);
    }

    response = this->DecodeFrame(nullptr, pCodecContext, pFrame);
    avformat_close_input(&pFormatCtx);
    avformat_free_context(pFormatCtx);
    av_packet_free(&pPacket);
    av_frame_free(&pFrame);
    avcodec_free_context(&pCodecContext);

    vector<FeatureMap> out;
}

FrameResponse Codecs::DecodeFrame(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame)
{
    
    FrameResponse frameResponse;

  // Supply raw packet data as input to a decoder
  // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga58bc4bf1e0ac59e27362597e467efff3
  frameResponse.response = avcodec_send_packet(pCodecContext, pPacket);
  if (frameResponse.response < 0) {
    cout<<"Error while sending a packet to the decoder"<<endl;
    exit(1);
  }

  while (frameResponse.response >= 0)
  {
    // Return decoded output data (into a frame) from a decoder
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga11e6542c4e66d3028668788a1a74217c
    frameResponse.response = avcodec_receive_frame(pCodecContext, pFrame);
    if (frameResponse.response == AVERROR(EAGAIN) || frameResponse.response == AVERROR_EOF) {
      break;
    } 
    else if (frameResponse.response < 0) 
    {
      cout<<"Error while receiving a frame from the decoder: %s"<<endl;
    //   return response;
        exit(1);
    }
    
    if (frameResponse.response >= 0) 
    {
        cout<<"Frame "<<pCodecContext->frame_number<<endl;
    //   cout<<"Pict Type"<<av_get_picture_type_char(pFrame->pict_type)<<endl;
    //   cout<<"Pkt Size "<<pFrame->pkt_size<<endl;
    //   cout<<"Pts "<<pFrame->pts<<endl;
    //   cout<<"Key Frames "<<pFrame->coded_picture_number<<endl;

    char frame_filename[1024];
    snprintf(frame_filename, sizeof(frame_filename), "%s-%d.pgm", "frame", pCodecContext->frame_number);
    // save a grayscale frame into a .pgm file
    save_gray_frame(pFrame->data[0], pFrame->linesize[0], pFrame->width, pFrame->height, frame_filename);
    
    // add pFrame to frameResponse variable
    vector<int> buf; 
    for(int i =0 ;i<pFrame->height; i++)
        {
            for(int j=0; j<pFrame->width; j++)
            {
                buf.push_back(pFrame->data[0][j + i*pFrame->linesize[0]]);
            }
        }
    
    frameResponse.frames.push_back(FeatureMap(buf, pFrame->height, pFrame->width, q_min, q_max));

    // get pFrame into
    cout<<"LinSize :"<<pFrame->linesize[0]<<" "<<pFrame->linesize[1]<<" "<<pFrame->linesize[2]<<endl; 
    cout<<"Height :"<<pFrame->height<<" "<<pFrame->width<<endl; 
    
    vector<int> Y;
    
    cout<<"Y ";
    for(int i=0; i<pFrame->height; i++)
        {
            for(int j=0; j<pFrame->width; j++)
            {
                Y.push_back(pFrame->data[0][i*pFrame->linesize[0] + j]);
            }
        }

            
    // cout<<endl<<"U: "<<endl;
    // vector<double> U, V;
    // for(int i=0; i<pFrame->height/2 ; i++)
    //     {   
    //         for(int j=0; j<pFrame->width/2; j++)
    //         {
    //             U.push_back(pFrame->data[1][i*pFrame->linesize[1] + j]);
    //             // cout<<U[i*j]<<" ";
    //         }
    //     }

    av_frame_unref(pFrame);
    
    }
  }
  frameResponse.response=0;
  return frameResponse;
}


