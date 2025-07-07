# 🔊 Audio Classification: Identifying Environmental Sounds
📌 Overview
This project focuses on classifying environmental sounds using deep learning techniques. The model is trained on the UrbanSound8K dataset, which includes diverse real-world urban sounds such as dog barks, sirens, drilling, and more.

By converting audio clips into Mel-spectrograms, the project leverages Convolutional Neural Networks (CNNs) to learn spatial patterns and classify short (≤ 4s) sound snippets into one of 10 categories.

🎯 Goals
Preprocess raw audio data and extract features

Train a CNN model for multi-class sound classification

Evaluate performance using UrbanSound8K folds

Test the trained model on external/unseen audio files to assess generalization (included in notebook)

📁 Dataset
UrbanSound8K

8,732 short audio clips across 10 environmental classes:

Air Conditioner, Car Horn, Children Playing, Dog Bark, Drilling, Engine Idling, Gun Shot, Jackhammer, Siren, Street Music

Each clip is ≤ 4 seconds and comes with metadata (fold, class, file name)

Download UrbanSound8K

🧠 Model Architecture
Input: Mel-spectrogram (2D image representation of audio)

Base Model: CNN built with Keras

Layers: Conv2D, MaxPooling, Dropout, Dense

Output: Softmax layer for 10 sound classes

🧪 External Audio Testing
✅ The model was evaluated not only on test folds of UrbanSound8K, but also on external audio clips to test generalization.
📓 You can find the predictions on new audio files in the Jupyter notebook (project3(voice_recognition).ipynb).

📊 Results
High accuracy on validation set using stratified folds

Effective generalization to new, real-world audio clips

Visualizations include spectrograms, confusion matrices, and prediction confidence

📜 Suggested Papers
Environmental Sound Classification with CNNs
Explores the use of convolutional models with spectrogram input for classifying short sound clips
arXiv Link

Deep Learning for Audio Classification
Overview of CNNs, RNNs, and hybrid approaches for audio classification tasks
arXiv Link

🛠️ Technologies
Python, Keras, TensorFlow

Librosa for audio processing

Scikit-learn, NumPy, Matplotlib

Jupyter Notebook


