#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <torch/nn/options/pooling.h>
#pragma warning(disable : 4267)
#include <torch/torch.h>
#pragma warning(default : 4267)

struct NetImpl : torch::nn::Module {
    NetImpl() {
        conv1 = register_module(
            "conv1",
            torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)),
                                  torch::nn::ReLU(), torch::nn::BatchNorm2d(64), torch::nn::MaxPool2d(2)));

        conv2 = register_module(
            "conv2",
            torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 512, 3).padding(1)),
                                  torch::nn::ReLU(), torch::nn::BatchNorm2d(512), torch::nn::MaxPool2d(2)));

        conv3 = register_module(
            "conv3",
            torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
                                  torch::nn::ReLU(), torch::nn::BatchNorm2d(512), torch::nn::MaxPool2d(2)));

        fc1 =
            register_module("fc1", torch::nn::Sequential(torch::nn::Flatten(), torch::nn::Linear(512 * 2 * 2, 2)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = conv2->forward(x);
        x = conv3->forward(x);
        x = conv3->forward(x);
        x = conv3->forward(x);
        x = conv3->forward(x);
        x = fc1->forward(x);

        return torch::log_softmax(x, 1);
    }

    torch::nn::Sequential conv1 { nullptr };
    torch::nn::Sequential conv2 { nullptr };
    torch::nn::Sequential conv3 { nullptr };
    torch::nn::Sequential fc1 { nullptr };
};

TORCH_MODULE(Net);

#endif