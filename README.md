# Image_classification_AI
This project demonstrates an image classification pipeline built in Google Colab using:

***Custom Convolutional Neural Network (CNN)

*Transfer Learning with MobileNetV2**

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

notebooks/ → Jupyter notebook with preprocessing, training, and evaluation.(https://github.com/Nandhanabichu/image_classification_AI/blob/main/image_classification.ipynb)

models/ → Trained model files for both CNN and MobileNetV2.

visualizations/ → Loss/accuracy curves, confusion matrices, F1-plot and ROC curves.

README.md → Project summary, usage instructions, and results.


**📊 Results Summary:**

| Model       | Accuracy | Precision | Recall | F1-Score |
| ----------- | -------- | --------- | ------ | -------- |
| Custom CNN  | 75%      | 71%       | 85%    | 78%      |
| MobileNetV2 | 88%      | 93%       | 52%    | 67%      |



**📸 Visualizations**
1.CNN confusion matrix(https://github.com/Nandhanabichu/image_classification_AI/blob/main/cnn_confusion_matrix.png)

2.MobileNetV2 confusion matrix(https://github.com/Nandhanabichu/image_classification_AI/blob/main/mobilenetv2_confusion_matrix.png)

3.Accuracy/Loss comparison(https://github.com/Nandhanabichu/image_classification_AI/blob/main/accuracy-loss_comparison.png)

4.ROC curve(https://github.com/Nandhanabichu/image_classification_AI/blob/main/cnn_roc_curve.png)

5.F1-plot(https://github.com/Nandhanabichu/image_classification_AI/blob/main/F1-score_plot.png)



**🛠 Tech Stack:**

Python, NumPy, Pandas

TensorFlow/Keras

OpenCV, Matplotlib, Seaborn

Scikit-learn



