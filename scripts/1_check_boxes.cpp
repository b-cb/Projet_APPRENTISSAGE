#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

const std::vector<std::string> LABELS = {
    "dragon", "tigre", "chien", "rat", "ram", "horse", 
    "monkey", "bird", "ox", "serpent", "hare", "boar"
};

void check_yolo_labels(const std::string& img_dir, const std::string& lbl_dir) {
    std::vector<fs::path> image_paths;
    
    // Récupérer toutes les images
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            image_paths.push_back(entry.path());
        }
    }

    std::cout << image_paths.size() << " images trouvees. Appuyez sur n'importe quelle touche pour defiler. 'q' pour quitter.\n";

    for (const auto& img_path : image_paths) {
        cv::Mat img = cv::imread(img_path.string());
        if (img.empty()) continue;

        // Déduire le chemin du fichier .txt
        fs::path txt_path = fs::path(lbl_dir) / img_path.filename().replace_extension(".txt");

        if (fs::exists(txt_path)) {
            std::ifstream file(txt_path);
            int class_id;
            float xc, yc, w, h;
            while (file >> class_id >> xc >> yc >> w >> h) {
                // Conversion YOLO normalisée -> Pixels
                int x_min = static_cast<int>((xc - w/2) * img.cols);
                int y_min = static_cast<int>((yc - h/2) * img.rows);
                int x_max = static_cast<int>((xc + w/2) * img.cols);
                int y_max = static_cast<int>((yc + h/2) * img.rows);

                // Dessin
                cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 255, 0), 2);
                std::string label = (class_id < LABELS.size()) ? LABELS[class_id] : "ID:" + std::to_string(class_id);
                cv::putText(img, label, cv::Point(x_min, y_min - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Verification", img);
        char c = (char)cv::waitKey(0);
        if (c == 'q' || c == 27) break; // Quitter sur 'q' ou 'Echap'
    }
    cv::destroyAllWindows();
}

int main() {
    check_yolo_labels("../dataset/images", "../dataset/labels");
    return 0;
}
