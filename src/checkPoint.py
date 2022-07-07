"""## 3. Define the **improve checker** to save the best checkpoint"""


# Accuracy checker: mode "min" for loss, mode "max" for accuracy
class ImproveChecker():
    def __init__(self, mode='min', best_val=None):
        assert mode in ['min', 'max']
        self.mode = mode
        if best_val is not None:
            self.best_val = best_val
        else:
            if self.mode == 'min':
                self.best_val = np.inf
            elif self.mode == 'max':
                self.best_val = 0.0

    def _check(self, val):
        if self.mode == 'min':
            if val < self.best_val:
                print("[%s] Improved from %.4f to %.4f" % (self.__class__.__name__, self.best_val, val))
                self.best_val = val
                return True
            else:
                print("[%s] Not improved from %.4f" % (self.__class__.__name__, self.best_val))
                return False
        else:
            if val > self.best_val:
                print("[%s] Improved from %.4f to %.4f" % (self.__class__.__name__, self.best_val, val))
                self.best_val = val
                return True
            else:
                print("[%s] Not improved from %.4f" % (self.__class__.__name__, self.best_val))
                return False