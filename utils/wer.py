import numpy as np


def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : target index list
    h : output index list

    Returns
    -------
    int
    """
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint8)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    if len(r) == 0:
        return float(d[len(r)][len(h)])

    # mean normalized by length of target sequence
    return float(d[len(r)][len(h)]) / float(len(r))


def batch_wer(batch_size, r_index, r_value, h_index, h_value):
    '''
    wrap the wer function to cal the batch mean wer
    Args:
        batch_size: int
        r_index: reference sparse array index
        r_value: reference sparse array value
        h_index: hypothesis sparse array index
        h_value: hypothesis sparse array value

    Returns:
        mean wer : float

    '''
    batch_wers = []

    for batch_index in range(0, batch_size):
        r = []
        h = []
        # scan the sparse tensor, then generate
        # the reference sentence and hypothesis
        for value_index in range(0, len(r_value)):
            # choose the batch_index th reference
            if r_index[value_index][0] == batch_index:
                r.append(r_value[value_index])

        for value_index in range(0, len(h_value)):
            if h_index[value_index][0] == batch_index:
                h.append(h_value[value_index])

        # cal the wer for one pair(r, h)
        wer_value = wer(r, h)
        batch_wers.append(wer_value)
    return np.mean(batch_wers)


class WERCalculator(object):
    def __init__(self, ignore_label_list):
        self._ignore_label_set = set(ignore_label_list)

    def remove_residual(self, inputs):
        outputs = []
        for i in inputs:
            if i == -1:
                return np.asarray(outputs)
            if i in self._ignore_label_set:
                continue
            outputs.append(i)
        return np.asarray(outputs)

    def cal_batch_wer(self, batch_r, batch_h):
        wers = []
        batch_size = len(batch_r)
        for i in np.arange(batch_size):
            r = batch_r[i]
            r = self.remove_residual(r)
            if len(r) == 0:
                wers.append(0.)
                continue
            h = self.remove_residual(batch_h[i])
            wers.append(wer(r, h))

        return np.asarray(wers)

    def cal_topk_wers(self,
                      batch_r, batch_h,
                      batch_size, nums_gpu,
                      topk, max_topk):
        list_wers = []
        for gpu_index in np.arange(nums_gpu):
            wers = []
            r = batch_r[gpu_index * batch_size: (gpu_index + 1) * batch_size]
            h = batch_h[gpu_index * batch_size * max_topk:
            (gpu_index + 1) * batch_size * max_topk]
            for i in np.arange(topk):
                topk_h = h[i * batch_size: (i + 1) * batch_size]
                wers.append(self.cal_batch_wer(r, topk_h))
            topk_wers = np.min(np.vstack(wers), axis=0)
            list_wers.extend(topk_wers)

        return list_wers


