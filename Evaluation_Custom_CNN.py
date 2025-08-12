#Evaluation_Custom CNN

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = (model_cnn.predict(X_val) > 0.5).astype("int32")

print(classification_report(Y_val, y_pred, target_names=['Cat','Dog']))

cm = confusion_matrix(Y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat','Dog'], yticklabels=['Cat','Dog'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()