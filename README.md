**Bienvenue dans notre projet d'apprentissage**

Ce projet a pour but de créer un réseau de neurones qui permet de détecter les **mudras**, signes de mains utilisés dans l'anime naruto.

**Ce repo contient plusieurs dossier :**
  - dataset : contient toutes nos images et les boites associées.
  - dataset_yolo : contient la base de données  triée pour pouvoir entrainer notre réseau de neuronne.
  - scripts : contient des scripts qui permettent d'avoir des informations sur la base de données et de la scinder pour l'entrainement .
  - Test_YOLO : contient un script python et un environnement associé pour pouvoir tester le réseau directement via la webcam sur linux.

**Pour tester le réseau via la webcam**
Activer la bulle, depuis /Test_YOLO faire la commande
'''source env_yolo/bin/activate'''
Puis : '''python3 test_webcam.py'''
