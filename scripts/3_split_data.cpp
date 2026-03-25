#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <fstream>
#include <map>

namespace fs = std::filesystem;

// Fonction pour récupérer l'ID de la première classe dans le fichier YOLO
int get_dominant_class(const fs::path& txt_path) {
    if (!fs::exists(txt_path)) return -1; // -1 pour les images sans label (background)
    
    std::ifstream file(txt_path);
    int class_id = -1;
    // On lit juste le premier entier du fichier (qui correspond à l'ID de la classe)
    if (file >> class_id) {
        return class_id;
    }
    return -1; // Fichier vide
}

void split_dataset(const std::string& img_dir, const std::string& lbl_dir, const std::string& out_dir) {
    if (!fs::exists(img_dir)) {
        std::cerr << "Erreur: Dossier source introuvable.\n";
        return;
    }

    // Map pour regrouper les chemins d'images par ID de classe
    std::map<int, std::vector<fs::path>> class_to_images;

    // 1. Regrouper toutes les images par classe
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            fs::path img_path = entry.path();
            fs::path txt_path = fs::path(lbl_dir) / img_path.filename().replace_extension(".txt");
            
            int cls = get_dominant_class(txt_path);
            class_to_images[cls].push_back(img_path);
        }
    }

    // Préparation des dossiers de sortie
    std::vector<std::string> splits = {"train", "val", "test"};
    for (const auto& split : splits) {
        fs::create_directories(fs::path(out_dir) / "images" / split);
        fs::create_directories(fs::path(out_dir) / "labels" / split);
    }

    std::random_device rd;
    std::mt19937 g(rd());

    int total_train = 0, total_val = 0, total_test = 0;

    // 2. Faire le split 80/10/10 POUR CHAQUE classe indépendamment
    for (auto& [cls, images] : class_to_images) {
        // Mélange aléatoire au sein de la classe
        std::shuffle(images.begin(), images.end(), g);

        int n_total = images.size();
        int n_train = static_cast<int>(n_total * 0.80);
        int n_val = static_cast<int>(n_total * 0.10);
        
        // Les limites pour cette classe spécifique
        std::vector<std::pair<int, int>> bounds = {
            {0, n_train}, 
            {n_train, n_train + n_val}, 
            {n_train + n_val, n_total} // Le reste va dans le test
        };

        for (size_t i = 0; i < splits.size(); ++i) {
            std::string split_name = splits[i];

            for (int j = bounds[i].first; j < bounds[i].second; ++j) {
                fs::path img_path = images[j];
                fs::path txt_path = fs::path(lbl_dir) / img_path.filename().replace_extension(".txt");

                fs::path dest_img = fs::path(out_dir) / "images" / split_name / img_path.filename();
                fs::copy(img_path, dest_img, fs::copy_options::overwrite_existing);

                if (fs::exists(txt_path)) {
                    fs::path dest_txt = fs::path(out_dir) / "labels" / split_name / txt_path.filename();
                    fs::copy(txt_path, dest_txt, fs::copy_options::overwrite_existing);
                }
            }
            
            // Mise à jour des compteurs globaux
            if (i == 0) total_train += (bounds[i].second - bounds[i].first);
            if (i == 1) total_val += (bounds[i].second - bounds[i].first);
            if (i == 2) total_test += (bounds[i].second - bounds[i].first);
        }
    }

    std::cout << "Partitionnement stratifie termine avec succes !\n";
    std::cout << "Train : " << total_train << " images.\n";
    std::cout << "Val   : " << total_val << " images.\n";
    std::cout << "Test  : " << total_test << " images.\n";
}

int main() {
    split_dataset("../dataset/images", "../dataset/labels", "../dataset_yolo");
    return 0;
}