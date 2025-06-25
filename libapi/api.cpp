#include "api.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <algorithm>

// constants
constexpr float CONF_THRESHOLD = 0.5f;
constexpr float NMS_THRESHOLD  = 0.4f;
constexpr int   INP_WIDTH      = 640;
constexpr int   INP_HEIGHT     = 640;

static cv::dnn::Net net;

int yolo_init(const char* onnxPath) {
    try {
        net = cv::dnn::readNetFromONNX(onnxPath);
    } catch (...) {
        return -1;
    }
    return 0;
}

int yolo_detect(
    const char* imagePath,
    YoloBBox*   outArray,
    int         maxBoxes)
{
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty())
        return -1;

    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        1/255.0f,
        cv::Size(INP_WIDTH, INP_HEIGHT),
        cv::Scalar(),      // mean
        true,              // swapRB
        false,             // crop
        CV_32F             // output depth
    );
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    std::vector<int>       classIds;
    std::vector<float>     confidences;
    std::vector<cv::Rect>  boxes;

    const float* data    = reinterpret_cast<float*>(outs[0].data);
    int   rows           = outs[0].size[1];
    int   dims           = outs[0].size[2];
    float xFactor        = frame.cols  / static_cast<float>(INP_WIDTH);
    float yFactor        = frame.rows / static_cast<float>(INP_HEIGHT);

    for (int i = 0; i < rows; ++i) {
        float conf = data[i*dims + 4];
        if (conf < CONF_THRESHOLD) 
            continue;

        cv::Mat scores(1, dims-5, CV_32FC1, 
                       (void*)(data + i*dims + 5));
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
        if (maxClassScore < CONF_THRESHOLD) 
            continue;

        float cx = data[i*dims + 0];
        float cy = data[i*dims + 1];
        float w  = data[i*dims + 2];
        float h  = data[i*dims + 3];
        int left   = static_cast<int>((cx - 0.5f*w) * xFactor);
        int top    = static_cast<int>((cy - 0.5f*h) * yFactor);
        int width  = static_cast<int>(w * xFactor);
        int height = static_cast<int>(h * yFactor);

        classIds .push_back(classIdPoint.x);
        confidences.push_back(static_cast<float>(maxClassScore));
        boxes    .emplace_back(left, top, width, height);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(
        boxes, confidences,
        CONF_THRESHOLD, NMS_THRESHOLD,
        indices
    );

    int count = std::min(static_cast<int>(indices.size()), maxBoxes);
    for (int i = 0; i < count; ++i) {
        int idx = indices[i];
        auto&r = boxes[idx];
        outArray[i] = {
            r.x, r.y, r.width, r.height,
            classIds[idx],
            confidences[idx]
        };
    }
    return count;
}


void yolo_free() {
    net = cv::dnn::Net();
}