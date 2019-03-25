from utils import pairwise

name_list = [ 
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "pelvis",
        "neck",
        "top"
]

COLOR_MAP = {
    'instance': (143, 35, 35),
    'nose': (255, 0, 0),
    'right_shoulder': (79,143,35),
    'right_elbow': (106,255,0),
    'right_wrist': (191,255,0),
    'left_shoulder': (0,64,255),
    'left_elbow': (0,149,255),
    'left_wrist': (0,234,255),
    'right_hip': (35,98,143),
    'right_knee': (185, 215, 237),
    'right_ankle': (185,237,224),
    'left_hip': (107,35,143),
    'left_knee': (170,0,255),
    'left_ankle': (220,185,237),
    'right_eye': (255, 127, 0),
    'left_eye': (255, 212, 0),
    'right_ear': (231, 233, 185),
    'left_ear': (255, 255, 0),
    'pelvis': (255,0,170),
    'neck': (237,185,185),
    'top': (115, 115, 115)
}

EDGES_BY_NAME = [
    ['instance', 'neck'],
    ['instance', 'nose'],
    ['neck', 'nose'],
    ['neck', 'top'],
    ['neck', 'left_eye'],
    ['nose', 'left_eye'],
    ['left_eye', 'left_ear'],
    ['nose', 'right_eye'],
    ['neck', 'right_eye'],
    ['right_eye', 'right_ear'],
    ['instance', 'left_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['instance', 'right_shoulder'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['instance', 'pelvis'],
    ['instance', 'left_hip'],
    ['instance', 'right_hip'],
    ['pelvis', 'left_hip'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['pelvis', 'right_hip'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
]

KEYPOINT_NAMES = ['instance'] + name_list
EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'neck', 'left_eye', 'left_ear']
TRACK_ORDER_1 = ['instance', 'neck', 'right_eye', 'right_ear']
TRACK_ORDER_2 = ['instance', 'nose', 'left_eye', 'left_ear']
TRACK_ORDER_3 = ['instance', 'nose', 'right_eye', 'right_ear']
TRACK_ORDER_4 = ['instance', 'neck', 'top']
TRACK_ORDER_5 = ['instance', 'neck', 'nose']
TRACK_ORDER_6 = ['instance', 'left_shoulder', 'left_elbow', 'left_wrist']
TRACK_ORDER_7 = ['instance', 'right_shoulder', 'right_elbow', 'right_wrist']
TRACK_ORDER_8 = ['instance', 'pelvis', 'left_hip']
TRACK_ORDER_9 = ['instance', 'pelvis', 'right_hip']
TRACK_ORDER_10 = ['instance', 'left_hip', 'left_knee', 'left_ankle']
TRACK_ORDER_11 = ['instance', 'right_hip', 'right_knee', 'right_ankle']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4, TRACK_ORDER_5, TRACK_ORDER_6, TRACK_ORDER_7, TRACK_ORDER_8, TRACK_ORDER_9, TRACK_ORDER_10, TRACK_ORDER_11]

DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])

EPSILON = 1e-6

