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
    const unsigned char* bgr, int width, int height, int stride,
    YoloBBox* outArray, int maxBoxes)
{
    cv::Mat frame(height, width, CV_8UC3,
                  const_cast<unsigned char*>(bgr), stride);

    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        1/255.0f,
        cv::Size(INP_WIDTH, INP_HEIGHT),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );

    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    std::vector<int> ids;
    std::vector<float> confs;
    std::vector<cv::Rect> boxes;

    int count = std::min<int>(maxBoxes, static_cast<int>(boxes.size()));
    for (int i = 0; i < count; ++i) {
        const auto& r = boxes[i];
        outArray[i] = { r.x, r.y, r.width, r.height, ids[i], confs[i] };
    }
    return count;
}

void yolo_free() {
    net = cv::dnn::Net();
}