import argparse
import os
import time

import cv2
import functools
import itertools

from PIL import ImageDraw, Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from model import *
from config import *
from dataset import *
from aug import *

parser = argparse.ArgumentParser(description='PyTorch PoseProposalNet Testing')
parser.add_argument('--resume', default='', type=str, metavar='PATH', required=True,
                help='path to latest checkpoint (default: none)')
parser.add_argument("-imsize", "--image_size", default=384, help="set input image size")
parser.add_argument("-val", "--val_file", help="json file path")
parser.add_argument("-img", "--root_dir", help="path to train2017 or val2018")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


args = parser.parse_args()


# Load Model
model = models.__dict__['resnet18']()

# Detach under avgpoll layer in Resnet
modules = list(model.children())[:-2]
model = nn.Sequential(*modules)
model = PoseProposalNet(model)
print(model)

# Restore Model
if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    
    print("=> loaded checkpoint '{}' (epoch {})"
	  .format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))

model.cuda()

# Load Data
val_dataset = KeypointsDataset(json_file = args.val_file, root_dir = args.root_dir+"/val2017/",
	    transform=transforms.Compose([
		IAA(model.insize,'val'),
		ToTensor()
	    ]))

collate_fn = functools.partial(custom_collate_fn,
			    insize = model.insize,
			    outsize = model.outsize, 
			    keypoint_names = model.keypoint_names,
			    local_grid_size = model.local_grid_size,
			    edges = model.edges)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1, shuffle=False,
    num_workers=args.workers, collate_fn = collate_fn, pin_memory=True)

# Test
def test():
    model.eval()
    with torch.no_grad():

        for i, (target_img, delta, max_delta_ij, tx, ty, tw, th, te) in enumerate(val_loader):

            if i > 100:
                break
            img = target_img.cuda()
            delta = delta.cuda()
            max_delta_ij = max_delta_ij.cuda()
            tx = tx.cuda()
            ty = ty.cuda()
            tw = tw.cuda()
            th = th.cuda()
            te = te.cuda()

            # compute output
            feature_map = model(img).cpu()

            ## TODO
            K = len(model.keypoint_names)
            B, _, _, _ = target_img.shape
            outW, outH = model.outsize

            #loss function with torch
            resp = feature_map[:, 0 * K:1 * K, :, :]
            conf = feature_map[:, 1 * K:2 * K, :, :]
            x = feature_map[:, 2 * K:3 * K, :, :]
            y = feature_map[:, 3 * K:4 * K, :, :]
            w = feature_map[:, 4 * K:5 * K, :, :]
            h = feature_map[:, 5 * K:6 * K, :, :]
            e = feature_map[:, 6 * K:, :, :].reshape(
                B,
                len(model.edges),
                9, 9,
                outH, outW
            )
            
            resp = np.squeeze(resp.numpy(), axis=0)
            conf = np.squeeze(conf.numpy(), axis=0)
            x = np.squeeze(x.numpy(), axis=0)
            y = np.squeeze(y.numpy(), axis=0)
            w = np.squeeze(w.numpy(), axis=0)
            h = np.squeeze(h.numpy(), axis=0)
            e = np.squeeze(e.numpy(), axis=0) 

            humans = get_humans_by_feature(model, resp, conf, x, y, w, h, e)
            
            pil_image = Image.fromarray(np.squeeze(target_img.numpy(), axis=0).astype(np.uint8).transpose(1, 2, 0))

            pil_image = draw_humans(
                keypoint_names=model.keypoint_names,
                edges=model.edges,
                pil_image=pil_image,
                humans=humans
            )

            pil_image.save('output/test/predict_test_result_'+str(i)+'.png', 'PNG')
 
# Recover size
def restore_xy(x, y):
    gridW, gridH = model.gridsize
    outW, outH = model.outsize
    X, Y = np.meshgrid(np.arange(outW, dtype=np.float32), np.arange(outH, dtype=np.float32))
    return (x + X) * gridW, (y + Y) * gridH

def restore_size(w, h):
    inW, inH = model.insize
    return inW * w, inH * h
    
# Parse result
def get_humans_by_feature(model, resp, conf, x, y, w, h, e, detection_thresh=0.09, min_num_keypoints=-1):
    start = time.time()

    delta = resp * conf
    #K = len(KEYPOINT_NAMES)
    outW, outH = model.outsize
    ROOT_NODE = 0  # instance

    rx, ry = restore_xy(x, y)
    rw, rh = restore_size(w, h)
    ymin, ymax = ry - rh / 2, ry + rh / 2
    xmin, xmax = rx - rw / 2, rx + rw / 2
    bbox = np.array([ymin, xmin, ymax, xmax])
    bbox = bbox.transpose(1, 2, 3, 0)
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE]
    logger.info('score: %s', score.shape)
    # Find person boxes which better confidence than the threshold
    candidate = np.where(score > detection_thresh)
    logger.info('candidate: %s', candidate)
    
    score = score[candidate]

    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox, thresh=0.5, score=score)
    logger.info('selected: %s', selected)
    root_bbox = root_bbox[selected]
    logger.info('detect instance {:.5f}'.format(time.time() - start))

    start = time.time()

    humans = []
    
    logger.info('delta shape: %s', delta.shape)
    logger.info('x shape: %s', x.shape)
    logger.info('y shape: %s', y.shape)
    logger.info('w shape: %s', w.shape)
    logger.info('h shape: %s', h.shape)
    logger.info('e shape: %s', e.shape)
    e = e.transpose(0, 3, 4, 1, 2)
    logger.info('e shape: %s', e.shape)
    ei = 0  # index of edges which contains ROOT_NODE as begin
    # alchemy_on_humans
    for hxw in zip(candidate[0][selected], candidate[1][selected]):
        human = {ROOT_NODE: bbox[(ROOT_NODE, hxw[0], hxw[1])]}  # initial
        for graph in DIRECTED_GRAPHS:
            eis, ts = graph
            i_h, i_w = hxw
            for ei, t in zip(eis, ts):
                index = (ei, i_h, i_w)  # must be tuple
                u_ind = np.unravel_index(np.argmax(e[index]), e[index].shape)
                j_h = i_h + u_ind[0] - model.local_grid_size[1] // 2
                j_w = i_w + u_ind[1] - model.local_grid_size[0] // 2
                if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                    break
                if delta[t, j_h, j_w] < detection_thresh:
                    break
                human[t] = bbox[(t, j_h, j_w)]
                i_h, i_w = j_h, j_w
        if min_num_keypoints <= len(human) - 1:
            humans.append(human)
    logger.info('alchemy time {:.5f}'.format(time.time() - start))
    logger.info('num humans = {}'.format(len(humans)))
    return humans

# NMS
def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)

# Drawing result
def draw_humans(keypoint_names, edges, pil_image, humans, mask=None, visbbox=False):
    """
    This is what happens when you use alchemy on humans...
    note that image should be PIL object
    """
    start = time.time()
    drawer = ImageDraw.Draw(pil_image)
    for human in humans:
        for k, b in human.items():
            if mask:
                fill = (255, 255, 255) if k == 0 else None
            else:
                fill = None
            ymin, xmin, ymax, xmax = b
            if k == 0:  # human instance
                # adjust size
                t = 1
                xmin = int(xmin * t + xmax * (1 - t))
                xmax = int(xmin * (1 - t) + xmax * t)
                ymin = int(ymin * t + ymax * (1 - t))
                ymax = int(ymin * (1 - t) + ymax * t)
                if mask:
                    resized = mask.resize(((xmax - xmin), (ymax - ymin)))
                    pil_image.paste(resized, (xmin, ymin), mask=resized)
                else:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
            else:
                if visbbox:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
                else:
                    r = 2
                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2
                    drawer.ellipse((x - r, y - r, x + r, y + r),
                                   fill=COLOR_MAP[keypoint_names[k]])

        for s, t in edges:
            if s in human and t in human:
                by = (human[s][0] + human[s][2]) / 2
                bx = (human[s][1] + human[s][3]) / 2
                ey = (human[t][0] + human[t][2]) / 2
                ex = (human[t][1] + human[t][3]) / 2

                drawer.line([bx, by, ex, ey],
                            fill=COLOR_MAP[keypoint_names[s]], width=2)

    logger.info('draw humans {: .5f}'.format(time.time() - start))
    return pil_image

if __name__=='__main__':
    test()
