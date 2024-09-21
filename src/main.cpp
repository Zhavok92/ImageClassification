#include "ImageRecognition.h"

#include <windows.h>
#include <exception>

int main() {
    try {
        DataPaths paths { "../data/train", "../data/test" };
        ImageRecognition ir(torch::kCUDA, paths);
        ir.train("CatsVsDogs", 85.f, 15);
        // ir.loadModel("CatsVsDogs.pt");

        for(;;) {
            system("cls");
            std::cout << "Bild: ";
            std::string path {};
            std::getline(std::cin, path);

            auto result { ir.makePrediction(path) };
            if(result.first == 0) {
                std::cout << "Hund (" << result.second << "%)\n";
            } else {
                std::cout << "Katze (" << result.second << "%)\n";
            }
            std::cin.get();
        }
    } catch(std::exception& exc) {
        std::cerr << exc.what() << '\n';
    }
}
