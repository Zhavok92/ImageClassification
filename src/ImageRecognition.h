#ifndef IMAGE_RECOGNITION_H
#define IMAGE_RECOGNITION_H
#include "NeuralNetwork.h"

#include <memory>
#include <utility>

#include "tensorboard_logger.h"

struct DataPaths {
    std::string trainData {};
    std::string testData {};
};

class ImageRecognition {
public:
    ImageRecognition(const torch::Device& device, const DataPaths& paths);
    float train(const std::string& modelName, float minAccuracy, int maxEpochs);
    std::pair<float, float> test();
    void saveModel(const std::string& name);
    void loadModel(const std::string& name);
    std::pair<int, float> makePrediction(const std::string& path);

private:
    std::string createDirectories();
    void logTrainData(int epoch, float trainLoss, float testLoss, float testAccuracy);
    std::unique_ptr<TensorBoardLogger> logger {};
    torch::Device device;
    DataPaths dataPaths {};
    std::string modelPath {};
    Net model {};
};

#endif