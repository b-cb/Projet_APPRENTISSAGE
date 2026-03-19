# 🥷 Projet Apprentissage Profond N7 : Détection de Signes Naruto

Bienvenue sur le dépôt de notre projet d'Apprentissage Profond (4ème année ENSEEIHT). 
L'objectif est d'entraîner un modèle de détection d'objet (YOLO) à reconnaître en temps réel les 12 signes de mains de l'univers de Naruto.

⚠️ **Échéance Base de Données : 25 Mars 2026 à 23h59**
⚠️ **Rapport n°1 (5 pages) : 26 Mars 2026 à 23h59**

---

## 📂 1. Structure du Dépôt

Respectez **strictement** cette arborescence lors de vos `git push`. Ne pushez jamais vos images en vrac à la racine !

```text
Projet_APPRENTISSAGE/
├── dataset/             <-- Données brutes de tout le monde
│   ├── images/          <-- Placez TOUTES vos images .jpeg/.jpg ici
│   └── labels/          <-- Placez TOUS vos fichiers .txt (YOLO) ici
├── dataset_yolo/        <-- Dossier généré automatiquement pour l'entraînement (Train/Val/Test)
├── scripts/             <-- Outils C++ de vérification et de traitement
└── README.md
```

---

## 🏷️ 2. Guide d'Annotation avec CVAT

Nous utilisons **CVAT** en local via Docker pour annoter nos images.

### Lancer CVAT sur votre PC
1. Ouvrez un terminal dans le dossier d'installation de CVAT.
2. Lancez les conteneurs : `docker compose up -d` *(ajoutez `sudo` sur Ubuntu si besoin)*.
3. Allez sur votre navigateur à l'adresse : **`http://localhost:8080`**.

### Importer / Exporter les données
* **Créer la tâche :** Utilisez le fichier `backup.zip` fourni par le groupe, ou créez une tâche en utilisant **impérativement** le dictionnaire JSON du groupe (pour avoir les mêmes IDs de 0 à 11 configurés en "rectangle").
* **Exporter vos annotations :** Une fois votre lot d'images terminé, allez dans les options de la tâche -> *Export task dataset* -> Choisissez **YOLO 1.1** -> Cochez *Save images*.
* **Git Push :** Prenez les images et les `.txt` de l'archive téléchargée, et placez-les respectivement dans `dataset/images/` et `dataset/labels/`.

---

## 📜 3. Charte d'Annotation & Nommage (TRÈS IMPORTANT)

Pour que la fusion sur Git fonctionne et que notre modèle soit performant, appliquez ces 3 règles à la lettre :

1. **Nommage anti-crash :** Avant d'importer la moindre image sur CVAT, renommez vos fichiers avec votre prénom pour ne pas écraser le travail des autres.
   * ✅ *Bon :* `batiste_001.jpg`, `lucas_042.jpg`
   * ❌ *Mauvais :* `image_1.jpg`, `capture.png`
2. **La règle du "Tight Box" :** Le rectangle doit effleurer les extrémités des doigts et la base des mains. Presque **aucun espace vide** dans la boîte.
3. **Seulement les mains :** Si le signe cache un visage, la boîte englobe **uniquement les mains**. Le modèle ne doit pas apprendre à détecter nos têtes.

---

## 🛠️ 4. Scripts Utilitaires (C++)

Voici les codes sources de nos 3 outils de traitement. Ils nécessitent `OpenCV` pour fonctionner.
*Sous Ubuntu, installez les prérequis avec : `sudo apt install libopencv-dev g++ pkg-config`*

### Script 1 : Vérification visuelle (`1_check_boxes.cpp`)
**À quoi ça sert ?** Il scanne le dossier `dataset/` et affiche vos images avec les rectangles verts par-dessus. À faire obligatoirement avant un Git Push pour vérifier que vos labels ne sont pas décalés.
* **Compilation :** `g++ -std=c++17 1_check_boxes.cpp -o 1_check_boxes $(pkg-config --cflags --libs opencv4)`
* **Exécution :** `./1_check_boxes`

**Code source :**
```cpp
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
    
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            image_paths.push_back(entry.path());
        }
    }

    std::cout << image_paths.size() << " images trouvees. Appuyez sur une touche pour defiler. 'q' pour quitter.\n";

    for (const auto& img_path : image_paths) {
        cv::Mat img = cv::imread(img_path.string());
        if (img.empty()) continue;

        fs::path txt_path = fs::path(lbl_dir) / img_path.filename().replace_extension(".txt");

        if (fs::exists(txt_path)) {
            std::ifstream file(txt_path);
            int class_id;
            float xc, yc, w, h;
            while (file >> class_id >> xc >> yc >> w >> h) {
                int x_min = static_cast<int>((xc - w/2) * img.cols);
                int y_min = static_cast<int>((yc - h/2) * img.rows);
                int x_max = static_cast<int>((xc + w/2) * img.cols);
                int y_max = static_cast<int>((yc + h/2) * img.rows);

                cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 255, 0), 2);
                std::string label = (class_id < LABELS.size()) ? LABELS[class_id] : "ID:" + std::to_string(class_id);
                cv::putText(img, label, cv::Point(x_min, y_min - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Verification", img);
        char c = (char)cv::waitKey(0);
        if (c == 'q' || c == 27) break; 
    }
    cv::destroyAllWindows();
}

int main() {
    check_yolo_labels("../dataset/images", "../dataset/labels");
    return 0;
}
```

---

### Script 2 : Statistiques de la Base (`2_stats_dataset.cpp`)
**À quoi ça sert ?** Il compte le nombre d'images et la répartition par classe. **Indispensable pour remplir notre rapport Moodle.**
* **Compilation :** `g++ -std=c++17 2_stats_dataset.cpp -o 2_stats_dataset`
* **Exécution :** `./2_stats_dataset`

**Code source :**
```cpp
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
```

---

### Script 3 : Partitionnement Train/Val/Test (`3_split_data.cpp`)
**À quoi ça sert ?** Il prend toutes les données en vrac dans `dataset/` et crée une architecture parfaite pour YOLO (80% Train, 10% Val, 10% Test) dans le dossier `dataset_yolo/`.
* **Compilation :** `g++ -std=c++17 3_split_data.cpp -o 3_split_data`
* **Exécution :** `./3_split_data`

**Code source :**
```cpp
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
```

