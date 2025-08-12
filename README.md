# Image_classification_AI
This project demonstrates an image classification pipeline built in Google Colab using:

*Custom Convolutional Neural Network (CNN)

*Transfer Learning with MobileNetV2

The workflow includes data preprocessing, augmentation, model training, evaluation, and performance comparison between the two approaches.


**The workflow includes**:

**Data Preprocessing** — Loading dataset (via Keras), normalization, resizing, visualization.

**Custom CNN Model** — Implemented from scratch, trained on preprocessed images, evaluated with accuracy/loss curves.

**Data Augmentation** — Rotation, flipping, zoom, and shift transformations to improve generalization.

**Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Classification Report.

**Transfer Learning** — Fine-tuning MobileNetV2 on the same dataset and comparing results with the custom CNN.

**Visualizations** — Loss/accuracy plots, confusion matrices, and sample predictions.



**🚀 Features:**

1.End-to-end pipeline in a single notebook.

2.Clear performance comparison between custom CNN and MobileNetV2.

3.Visual results for easy interpretation.

4.Ready-to-run on Google Colab with GPU support.


📂 **Repository Contents:**

notebooks/ → Jupyter notebook with preprocessing, training, and evaluation.

models/ → Trained model files for both CNN and MobileNetV2.

visualizations/ → Loss/accuracy curves, confusion matrices, F1-plot and ROC curves.

README.md → Project summary, usage instructions, and results.


**📊 Results Summary:**

| Model       | Accuracy | Precision | Recall | F1-Score |
| ----------- | -------- | --------- | ------ | -------- |
| Custom CNN  | 75%      | 71%       | 85%    | 78%      |
| MobileNetV2 | 88%      | 93%       | 52%    | 67%      |





**🛠 Tech Stack:**

Python, NumPy, Pandas

TensorFlow/Keras

OpenCV, Matplotlib, Seaborn

Scikit-learn



