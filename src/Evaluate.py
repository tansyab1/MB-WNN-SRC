class Evaluate():
    def __init__(self, coefLL=None, coefLH=None, coefHL=None,
                 coefHH=None, train_labels=None, test_labels=None):
        self.coefLL = coefLL
        self.coefLH = coefLH
        self.coefHL = coefHL
        self.coefHH = coefHH

        self.train_labels = train_labels
        self.test_labels = test_labels

        self.class_LL = torch.zeros(int(torch.max(self.test_labels)))
        self.class_LH = torch.zeros(int(torch.max(self.test_labels)))
        self.class_HL = torch.zeros(int(torch.max(self.test_labels)))
        self.class_HH = torch.zeros(int(torch.max(self.test_labels)))

        self.prediction = torch.zeros(len((self.test_labels)))
        self.prediction2 = torch.zeros(len((self.test_labels)))
        self.prediction3 = torch.zeros(len((self.test_labels)))

    def _eval(self):
        CoefLL = torch.abs(self._get_threshold(self.coefLL.cpu()))
        CoefLH = torch.abs(self._get_threshold(self.coefLH.cpu()))
        CoefHL = torch.abs(self._get_threshold(self.coefHL.cpu()))
        CoefHH = torch.abs(self._get_threshold(self.coefHH.cpu()))

        for atom in range(0, len(self.test_labels)):
            xll = CoefLL[atom, :]
            xlh = CoefLH[atom, :]
            xhl = CoefHL[atom, :]
            xhh = CoefHH[atom, :]

            for l in range(1, torch.max(self.test_labels) + 1):
                l_idx = np.array([j for j in range(0, len(self.train_labels)) if self.train_labels[j] == l]).astype(int)
                self.class_LL[int(l - 1)] = sum(torch.abs(xll[l_idx]))
                self.class_LH[int(l - 1)] = sum(torch.abs(xlh[l_idx]))
                self.class_HL[int(l - 1)] = sum(torch.abs(xhl[l_idx]))
                self.class_HH[int(l - 1)] = sum(torch.abs(xhh[l_idx]))

            self.prediction[atom] = torch.argmax(0.6 * self.class_LL + 0.15  * self.class_LH+ 0.15 * self.class_HL + 1 / 10 * self.class_HH) + 1
            self.prediction2[atom] = torch.argmax(0.5 * self.class_LL + 0.2 * self.class_LH + 0.2 * self.class_HL + 1 / 10 * self.class_HH) + 1
            self.prediction3[atom] = torch.argmax(
                0.4 * self.class_LL + 0.2 * self.class_LH + 0.2 * self.class_HL +0.2 * self.class_HH) + 1

        # self.prediction = np.array(self.prediction)
        missrate = self._error_cal(self.test_labels, self.prediction)
        missrate2 = self._error_cal(self.test_labels, self.prediction2)
        missrate3 = self._error_cal(self.test_labels, self.prediction3)
        accuracy = 1 - missrate
        accuracy2 = 1 - missrate2
        accuracy3 = 1 - missrate3
        return accuracy,accuracy2,accuracy3

    def _error_cal(self, ground_truth, predicted_label):
        ground_truth = torch.squeeze(ground_truth, 1)
        _error_value = (ground_truth != predicted_label).sum().item()
        missrate = _error_value / (ground_truth.shape[0])
        return missrate

    def _get_threshold(self, coef, ro=0.1):
        if ro < 1:
            Cp = torch.zeros((coef.shape[0], coef.shape[1]))
            sorted, _ = torch.sort(-torch.abs(coef), 0)
            S = torch.abs(sorted)
            _index = torch.argsort(-torch.abs(coef), 0)
            for i in range(coef.shape[1]):
                cL1 = torch.sum(S[:, i], dtype=torch.float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[_index[0:t + 1, i], i] = coef[_index[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = coef

        return Cp
