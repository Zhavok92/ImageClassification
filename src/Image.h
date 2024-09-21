#ifndef IMAGE_H
#define IMAGE_H
#include <vector>

#pragma warning(disable : 4267)
#include <torch/torch.h>
#pragma warning(default : 4267)
#include <opencv2/opencv.hpp>

namespace Image {
cv::Mat cropCenter(const cv::Mat& image) {
    int rows { image.rows };
    int cols { image.cols };

    int cropSize { std::min(rows, cols) };
    int offsetW { (cols - cropSize) / 2 };
    int offsetH { (rows - cropSize) / 2 };
    cv::Rect rectOfImage { offsetW, offsetH, cropSize, cropSize };

    return image(rectOfImage);
}

torch::Tensor transform(const std::string& path) {
    std::vector<double> normMean { 0.485, 0.456, 0.406 };
    std::vector<double> normStd { 0.229, 0.224, 0.225 };

    cv::Mat image { cv::imread(path) };
    image = cropCenter(image);
    cv::resize(image, image, cv::Size(128, 128));

    if(image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    } else {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }

    image.convertTo(image, CV_32FC3, 1 / 255.0);

    torch::Tensor imageTensor { torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kFloat) };
    imageTensor = imageTensor.permute({ 2, 0, 1 });
    imageTensor = torch::data::transforms::Normalize<>(normMean, normStd)(imageTensor);

    return imageTensor;
}
}

#endif