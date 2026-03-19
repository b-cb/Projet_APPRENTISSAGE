#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

void split_dataset(const std::string& img_dir, const std::string& lbl_dir, const std::string& out_dir) {
    std::vector<fs::path> images;
    
    if (!fs::exists(img_dir)) {
        std::cerr << "Erreur: Dossier source introuvable.\n";
        return;
    }

    for (const auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            images.push_back(entry.path());
        }
    }

    // Mélange aléatoire avec le moteur Mersenne Twister
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(images.begin(), images.end(), g);

    int n_total = images.size();
    int n_train = static_cast<int>(n_total * 0.80);
    int n_val = static_cast<int>(n_total * 0.10);
    
    std::vector<std::string> splits = {"train", "val", "test"};
    std::vector<std::pair<int, int>> bounds = {
        {0, n_train}, 
        {n_train, n_train + n_val}, 
        {n_train + n_val, n_total}
    };

    for (size_t i = 0; i < splits.size(); ++i) {
        std::string split_name = splits[i];
        fs::create_directories(fs::path(out_dir) / "images" / split_name);
        fs::create_directories(fs::path(out_dir) / "labels" / split_name);

        for (int j = bounds[i].first; j < bounds[i].second; ++j) {
            fs::path img_path = images[j];
            fs::path txt_path = fs::path(lbl_dir) / img_path.filename().replace_extension(".txt");

            fs::path dest_img = fs::path(out_dir) / "images" / split_name / img_path.filename();
            fs::copy(img_path, dest_img, fs::copy_options::overwrite_existing);

            // Ne copier le label que s'il existe (pour éviter un plantage sur des images sans signe)
            if (fs::exists(txt_path)) {
                fs::path dest_txt = fs::path(out_dir) / "labels" / split_name / txt_path.filename();
                fs::copy(txt_path, dest_txt, fs::copy_options::overwrite_existing);
            }
        }
        std::cout << split_name << " : " << (bounds[i].second - bounds[i].first) << " images.\n";
    }
    std::cout << "Partitionnement termine avec succes !\n";
}

int main() {
    split_dataset("../dataset/images", "../dataset/labels", "../dataset_yolo");
    return 0;
}

