// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"
// #import <Accelerate/Accelerate.h>
#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#include <time.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"
// #import <Accelerate/Accelerate.h>
#include "ios_image_load.h"

UIImage* RunInferenceOnImage();

namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }
        
        int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return (int)ifs_.gcount();
        }
        
    private:
        std::ifstream ifs_;
    };
}  // namespace

@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (IBAction)getUrl:(id)sender {
    UIImage* inference_result = RunInferenceOnImage();
    UIImageView *imageView = [[UIImageView alloc] initWithImage:inference_result];
    imageView.frame = CGRectMake(80, 100, 256, 256);
    [self.view addSubview:imageView];
}

@end

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
    ::google::protobuf::io::CopyingInputStreamAdaptor stream(
                                                             new IfstreamInputStream(file_name));
    stream.SetOwnsCopyingStream(true);
    // TODO(jiayq): the following coded stream is for debugging purposes to allow
    // one to parse arbitrarily large messages for MessageLite. One most likely
    // doesn't want to put protobufs larger than 64MB on Android, so we should
    // eventually remove this and quit loud when a large protobuf is passed in.
    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}

UIImage* RunInferenceOnImage() {
    tensorflow::SessionOptions options;
    
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!session_status.ok()) {
        std::string status_string = session_status.ToString();
        return NULL;
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    LOG(INFO) << "Session created.";
    
    tensorflow::GraphDef tensorflow_graph;
    LOG(INFO) << "Graph created.";
    
    NSString* network_path = FilePathForResourceName(@"tensorflow_inception_graph", @"pb");
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    
    LOG(INFO) << "Creating session.";
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
         return NULL;
    }
    
    // Read the Grace Hopper image.
    NSString* image_path = FilePathForResourceName(@"fff9b3a5373f_16", @"jpg");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
                                                                  [image_path UTF8String], &image_width, &image_height, &image_channels);
    
    const int wanted_width = 256;
    const int wanted_height = 256;
    const int wanted_channels = 3;
    const float input_mean = 0.0f;
    const float input_std = 255.0f;
    assert(image_channels >= wanted_channels);
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({
        1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8* in = image_data.data();
    // tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
    float* out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    std::string input_layer = "X_train";
    std::string output_layer = "unet/classifier/Relu";
    std::vector<tensorflow::Tensor> outputs;
    time_t start = time(NULL);
    tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
                                                 {output_layer}, {}, &outputs);
    time_t stop = time(NULL);
    time_t cost =difftime(stop,start);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        tensorflow::LogAllRegisteredKernels();
        return NULL;
    }
    
    
    tensorflow::string status_string = run_status.ToString();
    
    tensorflow::Tensor* output = &outputs[0];
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
    Eigen::Aligned> prediction = output->flat<float>();
    
    size_t bytesPerRow = wanted_width * 4;
    uint32_t *ImageBuf = (uint32_t *)malloc(bytesPerRow * wanted_height);
    uint32_t *pCurPtr = ImageBuf;
    int pixelNum = wanted_width * wanted_height;
    for (int i = 0; i<pixelNum; i++, pCurPtr++) {
        uint8_t* ptr = (uint8_t*)pCurPtr;
        float a  = prediction(i*2);
        float b = prediction(i*2+1);
        if (a > b) {
            ptr[0] = 255;
            ptr[1] = 0;
            ptr[2] = 0;
            ptr[3] = 0;
        }
        else {
            ptr[0] = 255;
            ptr[1] = 255;
            ptr[2] = 255;
            ptr[3] = 255;
        }
    }
    CGDataProviderRef dataProvider = CGDataProviderCreateWithData(NULL, ImageBuf,  bytesPerRow * wanted_height, NULL);
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
    CGImageRef imageRef = CGImageCreate(wanted_width, wanted_height, 8, 32, bytesPerRow, colorSpaceRef,
                                        kCGImageAlphaLast | kCGBitmapByteOrder32Little, dataProvider,NULL, true, kCGRenderingIntentDefault);
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    NSString *path = [[NSHomeDirectory()stringByAppendingPathComponent:@"Documents"]stringByAppendingPathComponent:@"image.png"];
    [UIImagePNGRepresentation(image) writeToFile:path  atomically:YES];
    return image;
}
