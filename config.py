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
        "thorax",
        "pelvis",
        "neck",
        "top"
]

COLOR_MAP = {
    'instance': (225, 225, 225),
    'nose': (255, 0, 0),
    'right_shoulder': (255, 170, 0),
    'right_elbow': (255, 255, 0),
    'right_wrist': (170, 255, 0),
    'left_shoulder': (85, 255, 0),
    'left_elbow': (0, 127, 0),
    'left_wrist': (0, 255, 85),
    'right_hip': (0, 170, 170),
    'right_knee': (0, 255, 255),
    'right_ankle': (0, 170, 255),
    'left_hip': (0, 85, 255),
    'left_knee': (0, 0, 255),
    'left_ankle': (85, 0, 255),
    'right_eye': (170, 0, 255),
    'left_eye': (255, 0, 255),
    'right_ear': (255, 0, 170),
    'left_ear': (255, 0, 85),
    'thorax': (128, 0, 80),
    'pelvis': (128, 80, 170),
    'neck': (255, 85, 0),
    'top': (200, 80, 0)
}

EDGES_BY_NAME = [
    ['instance', 'thorax'],
    ['thorax','neck'],
    ['neck', 'nose'],
    ['nose', 'top'],
    ['nose', 'left_eye'],
    ['left_eye', 'left_ear'],
    ['nose', 'right_eye'],
    ['right_eye', 'right_ear'],
    ['instance', 'left_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['instance', 'right_shoulder'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['instance', 'pelvis'],
    ['pelvis', 'left_hip'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['pelvis', 'right_hip'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
]

KEYPOINT_NAMES = ['instance'] + name_list
EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'thorax', 'neck', 'nose', 'left_eye', 'left_ear']
TRACK_ORDER_1 = ['instance', 'thorax', 'neck', 'nose', 'right_eye', 'right_ear']
TRACK_ORDER_2 = ['instance', 'thorax', 'neck', 'nose', 'top']
TRACK_ORDER_3 = ['instance', 'left_shoulder', 'left_elbow', 'left_wrist']
TRACK_ORDER_4 = ['instance', 'right_shoulder', 'right_elbow', 'right_wrist']
TRACK_ORDER_5 = ['instance', 'pelvis', 'left_hip', 'left_knee', 'left_ankle']
TRACK_ORDER_6 = ['instance', 'pelvis', 'right_hip', 'right_knee', 'right_ankle']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4, TRACK_ORDER_5, TRACK_ORDER_6]

DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])

EPSILON = 1e-6

