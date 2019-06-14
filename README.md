# MovementDetector_FaceRecognizer

1. Importy
- pip install opencv-python
- pip install imutils

2. Dodać do katalogu 'dataset' podkatalog ze zdjęciami osoby, którą ma rozpoznawać system, nazwać katalog jej imieniem.

3. Wywołać w konsoli kolejno:

- python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

- python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

- python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

DOKUMENTACJA: https://docs.google.com/document/d/1jIzDGRr_-HJ9-Y8GwxF9i-eJSGID_9zIEwQM0Y3vUaM/edit?usp=sharing
