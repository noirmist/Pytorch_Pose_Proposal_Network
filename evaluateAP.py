import numpy as np
import json
import os
import sys

import eval_helpers
from eval_helpers import Joint

def computeMetrics(scoresAll, labelsAll, nGTall):
    apAll = np.zeros((nGTall.shape[0] + 1, 1))
    recAll = np.zeros((nGTall.shape[0] + 1, 1))
    preAll = np.zeros((nGTall.shape[0] + 1, 1))
    # iterate over joints
    for j in range(nGTall.shape[0]):
        scores = np.zeros([0, 0], dtype=np.float32)
        labels = np.zeros([0, 0], dtype=np.int8)
        # iterate over images
        for imgidx in range(nGTall.shape[1]):
            scores = np.append(scores, scoresAll[j][imgidx])
            labels = np.append(labels, labelsAll[j][imgidx])
        # compute recall/precision values
        nGT = sum(nGTall[j, :])
        precision, recall, scoresSortedIdxs = eval_helpers.computeRPC(scores, labels, nGT)
        if (len(precision) > 0):
            apAll[j] = eval_helpers.VOCap(recall, precision) * 100
            preAll[j] = precision[len(precision) - 1] * 100
            recAll[j] = recall[len(recall) - 1] * 100
    idxs = np.argwhere(~np.isnan(apAll[:nGTall.shape[0],0]))
    apAll[nGTall.shape[0]] = apAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(recAll[:nGTall.shape[0],0]))
    recAll[nGTall.shape[0]] = recAll[idxs, 0].mean()
    idxs = np.argwhere(~np.isnan(preAll[:nGTall.shape[0],0]))
    preAll[nGTall.shape[0]] = preAll[idxs, 0].mean()

    return apAll, preAll, recAll


def evaluateAP(gtFramesAll, prFramesAll, bSaveAll=True, bSaveSeq=False):

    distThresh = 0.5

    names = Joint().name
    names['17'] = 'total'

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)

    # compute average precision (AP), precision and recall per part
    apAll, preAll, recAll = computeMetrics(scoresAll, labelsAll, nGTall)
    if (bSaveAll):
        metrics = {'ap': apAll.flatten().tolist(), 'pre': preAll.flatten().tolist(), 'rec': recAll.flatten().tolist(),  'names': names}
        filename = './total_AP_metrics.json'
        eval_helpers.writeJson(metrics,filename)

    return apAll, preAll, recAll
