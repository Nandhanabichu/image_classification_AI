#F1-score Plot

from sklearn.metrics import f1_score

# Compute F1 per class
f1_cat = f1_score(Y_val, y_pred, pos_label=0)
f1_dog = f1_score(Y_val, y_pred, pos_label=1)

plt.bar(['Cat', 'Dog'], [f1_cat, f1_dog], color=['#4CAF50', '#2196F3'])
plt.ylim(0, 1)
plt.title('F1-score per Class (Custom CNN)')
plt.ylabel('F1-score')
for i, score in enumerate([f1_cat, f1_dog]):
    plt.text(i, score + 0.02, f"{score:.2f}", ha='center')
plt.show()
