#include "Image.h"

#include <filesystem>

#pragma warning(disable : 4267)
#include <torch/torch.h>
#pragma warning(default : 4267)
#include <opencv2/opencv.hpp>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
public:
    CustomDataset(const std::string& dogFolder, const std::string& catFolder) {
        for(const auto& entry: std::filesystem::directory_iterator(dogFolder)) {
            images.emplace_back(entry.path().string());
            labels.emplace_back(0);
        }
        for(const auto& entry: std::filesystem::directory_iterator(catFolder)) {
            images.emplace_back(entry.path().string());
            labels.emplace_back(1);
        }
    }

    torch::data::Example<> get(size_t index) override {
        torch::Tensor imageTensor { Image::transform(images[index]) };
        torch::Tensor labelTensor { torch::tensor(labels[index], torch::kLong) };

        return { imageTensor, labelTensor };
    }

    torch::optional<size_t> size() const override {
        return images.size();
    }

private:
    std::vector<std::string> images;
    std::vector<int> labels;
};