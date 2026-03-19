#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include <iomanip>

namespace fs = std::filesystem;

const std::vector<std::string> LABELS = {
    "dragon", "tigre", "chien", "rat", "ram", "horse", 
    "monkey", "bird", "ox", "serpent", "hare", "boar"
};

int main() {
    std::string lbl_dir = "../dataset/labels";
    std::map<int, int> class_counts;
    int total_boxes = 0;
    int total_files = 0;

    if (!fs::exists(lbl_dir)) {
        std::cerr << "Erreur: Le dossier " << lbl_dir << " n'existe pas.\n";
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(lbl_dir)) {
        if (entry.path().extension() == ".txt") {
            total_files++;
            std::ifstream file(entry.path());
            int class_id;
            float xc, yc, w, h;
            while (file >> class_id >> xc >> yc >> w >> h) {
                class_counts[class_id]++;
                total_boxes++;
            }
        }
    }

    std::cout << "=== STATISTIQUES DE LA BASE DE DONNEES ===\n";
    std::cout << "Total d'images annotees : " << total_files << "\n";
    std::cout << "Total de signes detectes : " << total_boxes << "\n\n";
    std::cout << "Repartition par classe :\n";

    for (size_t i = 0; i < LABELS.size(); ++i) {
        int count = class_counts[i];
        double percentage = (total_boxes > 0) ? (static_cast<double>(count) / total_boxes * 100.0) : 0.0;
        std::cout << "- " << std::left << std::setw(10) << LABELS[i] << " : " 
                  << count << " (" << std::fixed << std::setprecision(1) << percentage << "%)\n";
    }

    return 0;
}
