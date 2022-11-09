'''
Nundita Lead here

'''


import numpy as np
from sklearn.cross_decomposition import CCA

class Analysis():
    def __init__(self):
        self.sampleLength = 500
        self.freqs = [5,7,8,9]
        self.noHarmonics = 2

    def formHarmonics(self):
        timeVector = np.linspace(0, 2, self.sampleLength) # 0 to 2 in 500 steps
        sig_sin = []
        sig_cos = []
        targets = {}
        for freq in self.freqs:
            for harmonics in range(self.noHarmonics):
                sig_sin.append(np.sin(2 * np.pi * harmonics * freq * timeVector))
                sig_cos.append(np.cos(2 * np.pi * harmonics * freq * timeVector))
            targets[freq] = np.array(sig_sin + sig_cos).T 
        return targets

    def formatSample(self, signal):
        return np.array(signal)[:, np.newaxis]

    def performCCA(self, signal):
        eeg = self.formatSample(signal)
        targets = self.formHarmonics()
        scores = [] 
        for key in targets: #key is the freq
            sig_c, t_c = CCA(n_components=1).fit_transform(eeg, targets[key])
            scores.append(np.corrcoef(sig_c.T, t_c.T)[0, 1])
        return scores


