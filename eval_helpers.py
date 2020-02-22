import numpy as np
from shapely import geometry
import sys
import os
import json
import glob

MIN_SCORE = -9999
MAX_TRACK_ID = 10000

class Joint:
    def __init__(self):
        self.count = 17
        self.left_shoulder = 0
        self.right_shoulder = 1
        self.left_elbow = 2
        self.right_elbow = 3
        self.left_wrist = 4
        self.right_wrist = 5
        self.left_hip = 6
        self.right_hip = 7
        self.left_knee = 8
        self.right_knee = 9
        self.left_ankle = 10
        self.right_ankle = 11
        self.thorax = 12
        self.pelvis = 13
        self.neck = 14
        self.top = 15
        self.stomach = 16

        self.name = {}
        self.name[self.right_ankle]    = "right_ankle"
        self.name[self.right_knee]     = "right_knee"
        self.name[self.right_hip]      = "right_hip"
        self.name[self.right_shoulder] = "right_shoulder"
        self.name[self.right_elbow]    = "right_elbow"
        self.name[self.right_wrist]    = "right_wrist"
        self.name[self.left_ankle]     = "left_ankle"
        self.name[self.left_knee]      = "left_knee"
        self.name[self.left_hip]       = "left_hip"
        self.name[self.left_shoulder]  = "left_shoulder"
        self.name[self.left_elbow]     = "left_elbow"
        self.name[self.left_wrist]     = "left_wrist"
        self.name[self.pelvis]         = "pelvis"
        self.name[self.thorax]         = "thorax"
        self.name[self.neck]           = "neck"
        self.name[self.top]            = "top"
        self.name[self.stomach]        = "stomach"

        self.symmetric_joint = {}
        self.symmetric_joint[self.right_ankle]    = self.left_ankle
        self.symmetric_joint[self.right_knee]     = self.left_knee
        self.symmetric_joint[self.right_hip]      = self.left_hip
        self.symmetric_joint[self.right_shoulder] = self.left_shoulder
        self.symmetric_joint[self.right_elbow]    = self.left_elbow
        self.symmetric_joint[self.right_wrist]    = self.left_wrist
        self.symmetric_joint[self.left_ankle]     = self.right_ankle
        self.symmetric_joint[self.left_knee]      = self.right_knee
        self.symmetric_joint[self.left_hip]       = self.right_hip
        self.symmetric_joint[self.left_shoulder]  = self.right_shoulder
        self.symmetric_joint[self.left_elbow]     = self.right_elbow
        self.symmetric_joint[self.left_wrist]     = self.right_wrist
        self.symmetric_joint[self.pelvis]         = -1
        self.symmetric_joint[self.thorax]         = -1
        self.symmetric_joint[self.neck]           = -1
        self.symmetric_joint[self.top]            = -1
        self.symmetric_joint[self.stomach]        = -1


def getPointGTbyID(points,pidx):

    point = []
    for i in range(len(points)):
        if (points[i]["id"] != None and points[i]["id"][0] == pidx): # if joint id matches
            point = points[i]
            break

    return point


def getHeadSize(x1,y1,x2,y2):
    headSize = 0.6*np.linalg.norm(np.subtract([x2,y2],[x1,y1]));
    return headSize


def formatCell(val,delim):
    return "{:>5}".format("%1.2f" % val) + delim


def getHeader():
    strHeader = "&"
    strHeader += " Head &"
    strHeader += " Shou &"
    strHeader += " Elb  &"
    strHeader += " Wri  &"
    strHeader += " Hip  &"
    strHeader += " Knee &"
    strHeader += " Ankl &"
    strHeader += " Total%s" % ("\\"+"\\")
    return strHeader

def getCum(vals):
    cum = []; n = -1
    cum += [(vals[[Joint().top,                    Joint().neck],0].mean())]
    cum += [(vals[[Joint().right_shoulder,Joint().left_shoulder],0].mean())]
    cum += [(vals[[Joint().right_elbow,   Joint().left_elbow   ],0].mean())]
    cum += [(vals[[Joint().right_wrist,   Joint().left_wrist   ],0].mean())]
    cum += [(vals[[Joint().right_hip,     Joint().left_hip     ],0].mean())]
    cum += [(vals[[Joint().right_knee,    Joint().left_knee    ],0].mean())]
    cum += [(vals[[Joint().right_ankle,   Joint().left_ankle   ],0].mean())]
    for i in range(Joint().count,len(vals)):
        cum += [vals[i,0]]
    return cum


def getFormatRow(cum):
    row = "&"
    for i in range(len(cum)-1):
        row += formatCell(cum[i]," &")
    row += formatCell(cum[len(cum)-1],(" %s" % "\\"+"\\"))
    return row


def printTable(vals):

    cum = getCum(vals)
    row = getFormatRow(cum)
    header = getHeader()
    print(header)
    print(row)
    return header+"\n", row+"\n"

# compute recall/precision curve (RPC) values
def computeRPC(scores,labels,totalPos):
    precision = np.zeros(len(scores))
    recall    = np.zeros(len(scores))
    npos = 0;

    idxsSort = np.array(scores).argsort()[::-1]
    labelsSort = labels[idxsSort];

    for sidx in range(len(idxsSort)):
        if (labelsSort[sidx] == 1):
            npos += 1
        # recall: how many true positives were found out of the total number of positives?
        recall[sidx]    = 1.0*npos / totalPos
        # precision: how many true positives were found out of the total number of samples?
        precision[sidx] = 1.0*npos / (sidx + 1)

    return precision, recall, idxsSort


# compute Average Precision using recall/precision values
def VOCap(rec,prec):

    mpre = np.zeros([1,2+len(prec)])
    mpre[0,1:len(prec)+1] = prec
    mrec = np.zeros([1,2+len(rec)])
    mrec[0,1:len(rec)+1] = rec
    mrec[0,len(rec)+1] = 1.0

    for i in range(mpre.size-2,-1,-1):
        mpre[0,i] = max(mpre[0,i],mpre[0,i+1])

    i = np.argwhere( ~np.equal( mrec[0,1:], mrec[0,:mrec.shape[1]-1]) )+1
    i = i.flatten()

    # compute area under the curve
    ap = np.sum( np.multiply( np.subtract( mrec[0,i], mrec[0,i-1]), mpre[0,i] ) )

    return ap

def get_data_dir():
    dataDir = "./"
    return dataDir

def help(msg=''):
    sys.stderr.write(msg+'\n')
    exit()

def process_arguments(argv):

    mode = 'multi'

    if len(argv) > 3:
        mode     = str.lower(argv[3])
    elif len(argv)<3 or len(argv)>4:
        help()

    gt_file = argv[1]
    pred_file = argv[2]

    if not os.path.exists(gt_file):
        help('Given ground truth directory does not exist!\n')

    if not os.path.exists(pred_file):
        help('Given prediction directory does not exist!\n')

    return gt_file, pred_file, mode

def cleanupData(gtFramesAll,prFramesAll):

    # remove all GT frames with empty annorects and remove corresponding entries from predictions
    imgidxs = []

    for imgidx in range(len(gtFramesAll)):
        if (len(gtFramesAll[imgidx]["annorect"]) > 0):
            imgidxs += [imgidx]
    gtFramesAll = [gtFramesAll[imgidx] for imgidx in imgidxs]
    prFramesAll = [prFramesAll[imgidx] for imgidx in imgidxs]

    # remove all gt rectangles that do not have annotations
    for imgidx in range(len(gtFramesAll)):
        gtFramesAll[imgidx]["annorect"] = removeRectsWithoutPoints(gtFramesAll[imgidx]["annorect"])
        prFramesAll[imgidx]["annorect"] = removeRectsWithoutPoints(prFramesAll[imgidx]["annorect"])

    return gtFramesAll, prFramesAll


def removeIgnoredPointsRects(rects,polyList):

    ridxs = list(range(len(rects)))
    for ridx in range(len(rects)):
        points = rects[ridx]["annopoints"][0]["point"]
        pidxs = list(range(len(points)))
        for pidx in range(len(points)):
            pt = geometry.Point(points[pidx]["x"][0], points[pidx]["y"][0])
            bIgnore = False
            for poidx in range(len(polyList)):
                poly = polyList[poidx]
                if (poly.contains(pt)):
                    bIgnore = True
                    break
            if (bIgnore):
                pidxs.remove(pidx)
        points = [points[pidx] for pidx in pidxs]
        if (len(points) > 0):
            rects[ridx]["annopoints"][0]["point"] = points
        else:
            ridxs.remove(ridx)
    rects = [rects[ridx] for ridx in ridxs]
    return rects


def removeIgnoredPoints(gtFramesAll,prFramesAll):

    imgidxs = []
    for imgidx in range(len(gtFramesAll)):
        if ("ignore_regions" in gtFramesAll[imgidx].keys() and
            len(gtFramesAll[imgidx]["ignore_regions"]) > 0):
            regions = gtFramesAll[imgidx]["ignore_regions"]
            polyList = []
            for ridx in range(len(regions)):
                points = regions[ridx]["point"]
                pointList = []
                for pidx in range(len(points)):
                    pt = geometry.Point(points[pidx]["x"][0], points[pidx]["y"][0])
                    pointList += [pt]
                poly = geometry.Polygon([[p.x, p.y] for p in pointList])
                polyList += [poly]

            rects = prFramesAll[imgidx]["annorect"]
            prFramesAll[imgidx]["annorect"] = removeIgnoredPointsRects(rects,polyList)
            rects = gtFramesAll[imgidx]["annorect"]
            gtFramesAll[imgidx]["annorect"] = removeIgnoredPointsRects(rects,polyList)

    return gtFramesAll, prFramesAll


def rectHasPoints(rect):
    return (("annopoints" in rect.keys()) and
            (len(rect["annopoints"]) > 0 and len(rect["annopoints"][0]) > 0) and
            ("point" in rect["annopoints"][0].keys()))


def removeRectsWithoutPoints(rects):

    idxsPr = []
    for ridxPr in range(len(rects)):
        if (rectHasPoints(rects[ridxPr])):
                idxsPr += [ridxPr];
    rects = [rects[ridx] for ridx in idxsPr]
    return rects

def load_data(gtFramesAll, prFramesAll):

    gtFramesAll,prFramesAll = cleanupData(gtFramesAll,prFramesAll)

    gtFramesAll,prFramesAll = removeIgnoredPoints(gtFramesAll,prFramesAll)

    return gtFramesAll, prFramesAll


def writeJson(val,fname):
    with open(fname, 'w') as data_file:
        json.dump(val, data_file)


def assignGTmulti(gtFrames, prFrames, distThresh):
    assert (len(gtFrames) == len(prFrames))

    nJoints = Joint().count
    # part detection scores
    scoresAll = {}
    # positive / negative labels
    labelsAll = {}
    # number of annotated GT joints per image
    nGTall = np.zeros([nJoints, len(gtFrames)])
    for pidx in range(nJoints):
        scoresAll[pidx] = {}
        labelsAll[pidx] = {}
        for imgidx in range(len(gtFrames)):
            scoresAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.float32)
            labelsAll[pidx][imgidx] = np.zeros([0, 0], dtype=np.int8)

    # number of GT poses
    nGTPeople = np.zeros((len(gtFrames), 1))
    # number of predicted poses
    nPrPeople = np.zeros((len(gtFrames), 1))

    for imgidx in range(len(gtFrames)):
        # distance between predicted and GT joints
        dist = np.full((len(prFrames[imgidx]["annorect"]), len(gtFrames[imgidx]["annorect"]), nJoints), np.inf)
        # score of the predicted joint
        score = np.full((len(prFrames[imgidx]["annorect"]), nJoints), np.nan)
        # body joint prediction exist
        hasPr = np.zeros((len(prFrames[imgidx]["annorect"]), nJoints), dtype=bool)
        # body joint is annotated
        hasGT = np.zeros((len(gtFrames[imgidx]["annorect"]), nJoints), dtype=bool)

        idxsPr = []
        for ridxPr in range(len(prFrames[imgidx]["annorect"])):
            if (("annopoints" in prFrames[imgidx]["annorect"][ridxPr].keys()) and
                ("point" in prFrames[imgidx]["annorect"][ridxPr]["annopoints"][0].keys())):
                idxsPr += [ridxPr];
        prFrames[imgidx]["annorect"] = [prFrames[imgidx]["annorect"][ridx] for ridx in idxsPr]

        nPrPeople[imgidx, 0] = len(prFrames[imgidx]["annorect"])
        nGTPeople[imgidx, 0] = len(gtFrames[imgidx]["annorect"])
        # iterate over GT poses(person)
        for ridxGT in range(len(gtFrames[imgidx]["annorect"])):
            # GT pose(person)
            rectGT = gtFrames[imgidx]["annorect"][ridxGT]
            pointsGT = []
            if len(rectGT["annopoints"]) > 0:
                pointsGT = rectGT["annopoints"][0]["point"]
            # iterate over all possible body joints
            for i in range(nJoints):
                # GT joint in MPII format
                ppGT = getPointGTbyID(pointsGT, i)
                if len(ppGT) > 0:
                    hasGT[ridxGT, i] = True

        # iterate over predicted poses
        for ridxPr in range(len(prFrames[imgidx]["annorect"])):
            # predicted pose
            rectPr = prFrames[imgidx]["annorect"][ridxPr]
            pointsPr = rectPr["annopoints"][0]["point"]
            for i in range(nJoints):
                # predicted joint in MPII format
                ppPr = getPointGTbyID(pointsPr, i)
                if len(ppPr) > 0:
                    if not ("score" in ppPr.keys()):
                        # use minimum score if predicted score is missing
                        if (imgidx == 0):
                            print('WARNING: prediction score is missing. Setting fallback score={}'.format(MIN_SCORE))
                        score[ridxPr, i] = MIN_SCORE
                    else:
                        score[ridxPr, i] = ppPr["score"][0]
                    hasPr[ridxPr, i] = True

        if len(prFrames[imgidx]["annorect"]) and len(gtFrames[imgidx]["annorect"]):
            # predictions and GT are present
            # iterate over GT poses
            for ridxGT in range(len(gtFrames[imgidx]["annorect"])):
                # GT pose
                rectGT = gtFrames[imgidx]["annorect"][ridxGT]
                # compute reference distance as head size
                headSize = getHeadSize(rectGT["x1"][0], rectGT["y1"][0],
                                                    rectGT["x2"][0], rectGT["y2"][0])
                pointsGT = []
                if len(rectGT["annopoints"]) > 0:
                    pointsGT = rectGT["annopoints"][0]["point"]
                # iterate over predicted poses
                for ridxPr in range(len(prFrames[imgidx]["annorect"])):
                    # predicted pose
                    rectPr = prFrames[imgidx]["annorect"][ridxPr]
                    pointsPr = rectPr["annopoints"][0]["point"]

                    # iterate over all possible body joints
                    for i in range(nJoints):
                        # GT joint
                        ppGT = getPointGTbyID(pointsGT, i)
                        # predicted joint
                        ppPr = getPointGTbyID(pointsPr, i)
                        # compute distance between predicted and GT joint locations
                        if hasPr[ridxPr, i] and hasGT[ridxGT, i]:
                            pointGT = [ppGT["x"][0], ppGT["y"][0]]
                            pointPr = [ppPr["x"][0], ppPr["y"][0]]
                            dist[ridxPr, ridxGT, i] = np.linalg.norm(np.subtract(pointGT, pointPr)) / headSize

            dist = np.array(dist)
            hasGT = np.array(hasGT)

            # number of annotated joints
            nGTp = np.sum(hasGT, axis=1)
            match = dist <= distThresh
            pck = 1.0 * np.sum(match, axis=2)
            for i in range(hasPr.shape[0]):
                for j in range(hasGT.shape[0]):
                    if nGTp[j] > 0:
                        pck[i, j] = pck[i, j] / nGTp[j]

            # preserve best GT match only
            idx = np.argmax(pck, axis=1)
            val = np.max(pck, axis=1)
            for ridxPr in range(pck.shape[0]):
                for ridxGT in range(pck.shape[1]):
                    if (ridxGT != idx[ridxPr]):
                        pck[ridxPr, ridxGT] = 0
            prToGT = np.argmax(pck, axis=0)
            val = np.max(pck, axis=0)
            prToGT[val == 0] = -1

            # assign predicted poses to GT poses
            for ridxPr in range(hasPr.shape[0]):
                if (ridxPr in prToGT):  # pose matches to GT
                    # GT pose that matches the predicted pose
                    ridxGT = np.argwhere(prToGT == ridxPr)
                    assert(ridxGT.size == 1)
                    ridxGT = ridxGT[0,0]
                    s = score[ridxPr, :]
                    m = np.squeeze(match[ridxPr, ridxGT, :])
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if (hp[i]):
                            scoresAll[i][imgidx] = np.append(scoresAll[i][imgidx], s[i])
                            labelsAll[i][imgidx] = np.append(labelsAll[i][imgidx], m[i])

                else:  # no matching to GT
                    s = score[ridxPr, :]
                    m = np.zeros([match.shape[2], 1], dtype=bool)
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if (hp[i]):
                            scoresAll[i][imgidx] = np.append(scoresAll[i][imgidx], s[i])
                            labelsAll[i][imgidx] = np.append(labelsAll[i][imgidx], m[i])
        else:
            if not len(gtFrames[imgidx]["annorect"]):
                # No GT available. All predictions are false positives
                for ridxPr in range(hasPr.shape[0]):
                    s = score[ridxPr, :]
                    m = np.zeros([nJoints, 1], dtype=bool)
                    hp = hasPr[ridxPr, :]
                    for i in range(len(hp)):
                        if hp[i]:
                            scoresAll[i][imgidx] = np.append(scoresAll[i][imgidx], s[i])
                            labelsAll[i][imgidx] = np.append(labelsAll[i][imgidx], m[i])

        # save number of GT joints
        for ridxGT in range(hasGT.shape[0]):
            hg = hasGT[ridxGT, :]
            for i in range(len(hg)):
                nGTall[i, imgidx] += hg[i]


    return scoresAll, labelsAll, nGTall
