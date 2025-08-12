#Result Comparison

print("\nCustom CNN Val Acc:", history_cnn.history['val_accuracy'][-1])
print("MobileNetV2 Val Acc:", history_mobilenet.history['val_accuracy'][-1])

