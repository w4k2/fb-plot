import numpy as np
from functools import cached_property


def confusion_matrix(y_true, y_pred, n_classes=2):
    y = n_classes * y_true + y_pred
    y = np.bincount(y, minlength=(n_classes * n_classes))
    return y.reshape(n_classes, n_classes)


def zero_division(a, b):
    return np.divide(a, b, out=np.zeros(a.shape), where=b!=0)


class BinaryConfusionMatrix:
    def __init__(self, y_true, y_pred):
        self.CM = confusion_matrix(y_true, y_pred)

    @cached_property
    def TP(self):
        # True Positive
        return self.CM[..., 1, 1]

    @cached_property
    def FN(self):
        # False Negative
        return self.CM[..., 1, 0]

    @cached_property
    def FP(self):
        # False Positive
        return self.CM[..., 0, 1]

    @cached_property
    def TN(self):
        # True Negative
        return self.CM[..., 0, 0]

    @cached_property
    def AP(self):
        # Actual Positive
        return self.TP + self.FN

    @cached_property
    def AN(self):
        # Actual Negative
        return self.TN + self.FP

    @cached_property
    def PP(self):
        # Predicted Positive
        return self.TP + self.FP

    @cached_property
    def PN(self):
        # Predicted Negative
        return self.FN + self.TN

    @cached_property
    def TPR(self):
        # True Positive Rate
        # return self.TP / self.AP
        return zero_division(self.TP, self.AP)

    @cached_property
    def FNR(self):
        # False Negative Rate
        return 1 - self.TPR

    @cached_property
    def TNR(self):
        # True Negative Rate
        # return self.TN / self.AN
        return zero_division(self.TN, self.AN)

    @cached_property
    def FPR(self):
        # False Positive Rate
        return 1 - self.TNR

    @cached_property
    def PPV(self):
        # Positive Predictive Value
        # return self.TP / self.PP
        return zero_division(self.TP, self.PP)

    @cached_property
    def FDR(self):
        # False Discovery Rate
        return 1 - self.PPV

    @cached_property
    def NPV(self):
        # Negative Predictive Value
        # return self.TN / self.PN
        return zero_division(self.TN, self.PN)

    @cached_property
    def FOR(self):
        # False Ommision Rate
        return 1 - self.NPV

    @cached_property
    def PLR(self):
        # Positive Likehood Ratio
        # return self.TPR / self.FPR
        return zero_division(self.TPR, self.FPR)

    @cached_property
    def NLR(self):
        # Negative Likehood Ratio
        # return self.FNR / self.TNR
        return zero_division(self.FNR, self.TNR)

    @cached_property
    def DOR(self):
        # Diagnostic Odds Ratio
        # return self.PLR / self.NLR
        return zero_division(self.PLR, self.NLR)

    @cached_property
    def total_population(self):
        return self.AP + self.AN

    @cached_property
    def prevalence(self):
        return self.AP / self.total_population

    @cached_property
    def markednes(self):
        return self.PPV + self.NPV - 1

    @cached_property
    def accuracy(self):
        return (self.TP + self.TN) / self.total_population

    @cached_property
    def balanced_accuracy(self):
        return (self.TPR + self.TNR) / 2

    @cached_property
    def f1_score(self):
        return (2 * self.PPV * self.TPR) / (self.PPV + self.TPR)

    @cached_property
    def gmean(self):
        return np.sqrt(self.TPR * self.TNR)

    @cached_property
    def fowlkes_mallows_index(self):
        return np.sqrt(self.PPV * self.TPR)

    @cached_property
    def matthews_correlation_coefficient(self):
        return np.sqrt(
            (self.TPR * self.TNR * self.PPV * self.NPV)
            - (self.FNR * self.FPR * self.FOR * self.FDR)
        )

    @cached_property
    def jaccard_index(self):
        return self.TP / (self.TP + self.FN + self.FP)

    # Legacy definitions
    @property
    def sensitivity(self):
        return self.TPR

    @property
    def recall(self):
        return self.TPR

    @property
    def specificity(self):
        return self.TNR

    @property
    def precision(self):
        return self.PPV

    def f_beta_score(self, beta=1):
        beta_sqr = beta ^ 2
        return (1 + beta_sqr) * self.PPV * self.TPR / ((beta_sqr * self.PPV) + self.TPR)

    # Calculate multiple metrics
    def get_metrics(self, metrics):
        return np.array([getattr(self, metric) for metric in metrics])

    def __str__(self):
        return str(self.CM)
