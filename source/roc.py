from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

actual = [1,1,1,0,0,0]
predictions = [0.6,0.55,0.7,0.2,0.3,0.1]

FPR, TPR, thresholds = roc_curve(actual, predictions)
roc_auc = auc(FPR, TPR)

plt.title("ROC")
plt.plot(FPR, TPR, 'b', label=f'AUC = {roc_auc}')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()