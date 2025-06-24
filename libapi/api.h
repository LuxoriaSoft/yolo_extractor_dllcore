#pragma once
#ifdef _WIN32
#  define YOLO_API extern "C" __declspec(dllexport)
#else
#  define YOLO_API extern "C"
#endif

struct YoloBBox {
    int x, y, w, h;
    int classId;
    float confidence;
};

YOLO_API int  yolo_init(const char* onnxPath);
YOLO_API int  yolo_detect(
    const unsigned char* bgr, int width, int height, int stride,
    YoloBBox* outArray, int maxBoxes);
YOLO_API void yolo_free();
