import numpy as np
import scipy.stats as stats


class Classifier:
    # Class Variables
    _n_class = _p_m_s = None

    # Constructor
    def __init__(self, col_1, col_2, n_class):
        self._init_var(col_1, col_2, n_class)

    # Methods
    def _init_var(self, col_1, col_2, n_class):
        self._n_class = n_class

        assert len(col_1) == len(col_2)
        hmap = self._sort_cols(col_1, col_2)

        assert self._n_class == len(list(hmap))
        self._load_prior(col_2)
        self._load_mean_std(hmap)

    def _load_prior(self, col_2):
        self._p_m_s = {}
        for i in range(self._n_class):
            self._p_m_s[i] = {"prior": col_2.count(i) / float(len(col_2))}

        return

    def _sort_cols(self, col_1, col_2):
        hmap = {}

        for i in range(len(col_1)):
            if col_2[i] not in hmap:
                hmap[col_2[i]] = []

            hmap[col_2[i]].append(col_1[i])

        return hmap

    def _load_mean_std(self, hmap):
        for k in list(hmap):
            self._p_m_s[k]["mean"] = np.mean(hmap[k])
            self._p_m_s[k]["std"] = np.std(hmap[k], ddof=1)

        return

    def classify(self, test_x):
        def likelihood_x_prior(x, class_n):
            pms = self._p_m_s[class_n]
            return stats.norm(pms["mean"], pms["std"]).pdf(x) * pms["prior"]

        evidence = 0

        for k in list(self._p_m_s):
            evidence += likelihood_x_prior(test_x, k)

        hmap = {}

        for k in list(self._p_m_s):
            if evidence != 0:
                post = likelihood_x_prior(test_x, k) / evidence
            else:
                post = 0

            if post not in hmap:
                hmap[post] = []

            hmap[post].append(k)

        class_list = hmap[np.max(list(hmap))]
        return class_list[np.random.randint(0, len(class_list))]
