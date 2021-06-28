import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class ConfusionMatrix:
    def __init__(self):
        pass

    def plot(self, cm, classes, title, fig, subplot_index):    
        normalize=True
        cmap=plt.cm.Blues

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax = fig.add_subplot(2,2,subplot_index)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax


class AUC:
    def __init__(self):
        pass

    def plot(self, y_test, y_pred, title='AUC', subplot_index=1):

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        lw = 2

        plt.subplot(2, 2, subplot_index)
        plt.title(title)
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
    
        return roc_auc