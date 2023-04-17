# Reconstruction3D
Projet de reconstruction 3D à l'aide d'une caméra stéréo

## Voici les versions des différentes technologies pour faire fonctionner le projet. 
S'assurer d'avoir les bonnes version avant de rouler le projet.
python -> 3.7.1
numpy -> 1.21.6
pip -> 23.0.1
opencv -> 4.5.1

## Important d'installer les bibliothèques ci-dessous avant de rouler le projet:
- opencv-python (4.5.1)
- numpy (1.21.6)
- mediapipe

Le fichier Main.py roule le projet pour une seule image que vous pouvez spécifier directement dans la ligne de commande.
Voici un exemple pour lancer le projet avec l'image Center38cm.png
```
python Main.py -f StereoImages/Center38cm.png
```

le fichier RealTime.py commence le programme pour faire l'acquisition en temps réel.
Il n'est donc pas nécessaire de mentionner de ficher 
```
python RealTime.py
```
