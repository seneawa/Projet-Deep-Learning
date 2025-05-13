#  Projet de Reconnaissance Faciale avec MLP (PyTorch)

##  Description

Ce projet implémente un système de **reconnaissance faciale** basé sur un **Perceptron Multicouche (MLP)** utilisant **PyTorch**.  
L'objectif est de classer correctement des visages d'individus à partir d'un ensemble d’images, après extraction des caractéristiques à l’aide de la méthode `face_recognition`.

---

##  Objectifs

- Extraire les **encodages faciaux** d’images avec la librairie `face_recognition`
- Créer un jeu de données labellisé à partir des visages extraits
- Construire un **MLP** en PyTorch pour la classification des visages
- Entraîner et évaluer le modèle sur les données préparées

---

##  Technologies utilisées

- **Python 3**
- **PyTorch**
- face_recognition
- NumPy
- Scikit-learn
- Matplotlib
- PIL

---

##  Méthodologie

1.  **Chargement des images** à partir de dossiers (chaque sous-dossier = une personne)
2.  **Encodage des visages** avec `face_recognition.face_encodings()`
3.  **Création du dataset** (X = encodages, y = labels)
4.  **Définition du MLP** avec 3 couches :
   - Entrée → 128 neurones → ReLU  
   - 64 neurones → ReLU  
   - Sortie (Softmax)
      **Fonction de perte** : `CrossEntropyLoss`  
      **Optimiseur** : `Adam`
      **Entraînement** du modèle pendant plusieurs époques
      **Évaluation** : taux de précision et matrice de confusion








