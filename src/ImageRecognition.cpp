#include "ImageRecognition.h"
#include "CustomDataSet.h"
#include "Image.h"

#include <chrono>
#include <random>
#include <sstream>
#include <string>

#include <torch/data/samplers/random.h>

ImageRecognition::ImageRecognition(const torch::Device& newDevice, const DataPaths& dataPaths)
    : device { newDevice }, dataPaths { dataPaths } {
    if((device == torch::kCUDA) && (!torch::cuda::is_available())) {
        std::cerr << "CUDA is not available. The program is executed on the CPU.\n";
        device = torch::Device(torch::kCPU);
    }
    model->to(device);
    modelPath = createDirectories();
    logger = std::make_unique<TensorBoardLogger>(modelPath + "tfevents.pb");
    std::size_t total_params {};
    for(const auto& param: model->parameters()) {
        total_params += param.numel();
    }

    std::stringstream modelStructure;
    modelStructure << model << '\n';
    logger->add_text("Model Structure: ", 0, modelStructure.str().c_str());
    logger->add_text("Model Parameters: ", 0, std::to_string(total_params).c_str());
}

float ImageRecognition::train(const std::string& modelName, float minAccuracy, int maxEpochs) {
    auto start { std::chrono::high_resolution_clock::now() };
    torch::optim::Adam optimizer { model->parameters(), 0.001 };

    auto dataset { CustomDataset(dataPaths.trainData + "/dogs", dataPaths.trainData + "/cats")
                       .map(torch::data::transforms::Stack<>()) };
    auto dataLoader { torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset),
                                                                                          32) };
    float currentAccuracy {};
    float bestAccuracy {};
    for(int epoch { 1 }; epoch <= maxEpochs; ++epoch) {
        float trainLoss {};
        for(auto& batch: *dataLoader) {
            model->train();
            auto images { batch.data.to(device) };
            auto labels { batch.target.to(device) };

            optimizer.zero_grad();
            auto output { model->forward(images) };
            auto loss { torch::nn::functional::cross_entropy(output, labels) };
            trainLoss = loss.item<float>();

            loss.backward();
            optimizer.step();
        }

        std::chrono::duration<double> duration { std::chrono::high_resolution_clock::now() - start };
        std::cout << "Epoch: " << epoch << " | Loss: " << trainLoss << " | " << duration.count() << "s";
        auto testData { test() };
        currentAccuracy = testData.second;
        logTrainData(epoch, trainLoss, testData.first, testData.second);

        if((currentAccuracy > minAccuracy) && (currentAccuracy > bestAccuracy)) {
            saveModel(modelPath + modelName + "_" + std::to_string(currentAccuracy) + ".pt");
            bestAccuracy = currentAccuracy;
        }
    }

    return currentAccuracy;
}

void ImageRecognition::logTrainData(int epoch, float trainLoss, float testLoss, float testAccuracy) {
    logger->add_scalar("Loss/Train", epoch, trainLoss);
    logger->add_scalar("Loss/Test", epoch, testLoss);
    logger->add_scalar("TestAccuracy", epoch, testAccuracy);

    for(const auto& named_param: model->named_parameters()) {
        std::string name { named_param.key() };
        torch::Tensor param { named_param.value().cpu() };

        if(name.find("weight") != std::string::npos) {
            std::vector<float> data(param.data_ptr<float>(), param.data_ptr<float>() + param.numel());
            logger->add_histogram("weights/" + name, epoch, data);
        }
        if(name.find("bias") != std::string::npos) {
            std::vector<float> data(param.data_ptr<float>(), param.data_ptr<float>() + param.numel());
            logger->add_histogram("biases/" + name, epoch, data);
        }
    }
}

std::pair<float, float> ImageRecognition::test() {
    auto dataset { CustomDataset(dataPaths.testData + "/dogs", dataPaths.testData + "/cats")
                       .map(torch::data::transforms::Stack<>()) };
    auto dataLoader { torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset),
                                                                                              32) };

    model->eval();
    float testLoss {};
    float testAccuracy {};
    int totalSamples {};
    for(const auto& batch: *dataLoader) {
        auto data { batch.data.to(device) };
        auto labels { batch.target.to(device) };
        auto output { model->forward(data) };

        auto loss { torch::nn::functional::cross_entropy(output, labels) };
        testLoss += loss.item<float>();

        auto predicted { output.argmax(1) };
        testAccuracy += predicted.eq(labels).sum().item<int>();

        totalSamples += labels.size(0);
    }

    float avgAcc { (testAccuracy / totalSamples) * 100 };

    std::cout << " | Test Loss: " << (testLoss / totalSamples)
              << " | Test Accuracy: " << (testAccuracy / totalSamples) * 100 << "%\n";

    return std::make_pair(testLoss, avgAcc);
}

void ImageRecognition::saveModel(const std::string& name) {
    torch::save(model, name);
}

void ImageRecognition::loadModel(const std::string& name) {
    torch::load(model, name);
}

std::pair<int, float> ImageRecognition::makePrediction(const std::string& path) {
    auto imageTensor { Image::transform(path) };
    imageTensor.unsqueeze_(0);
    imageTensor = imageTensor.to(device);
    model->eval();
    auto output { model->forward(imageTensor) };
    auto probabilities { torch::softmax(output, 1) };
    auto predicted { output.argmax(1).item<int>() };
    float confidence { probabilities[0][predicted].item<float>() * 100 };

    return std::make_pair(predicted, confidence);
}

std::string ImageRecognition::createDirectories() {
    std::string path { "models" };
    if(!std::filesystem::exists(path)) {
        std::filesystem::create_directory(path);
    }
    auto now { std::chrono::system_clock::now() };
    std::time_t nowC { std::chrono::system_clock::to_time_t(now) };
    std::tm nowTm {};
    localtime_s(&nowTm, &nowC);
    std::ostringstream oss;
    oss << std::put_time(&nowTm, "%Y-%m-%d_%H-%M-%S");
    path = path + "/" + oss.str();
    std::filesystem::create_directory(path);

    return path + "/";
}