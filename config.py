from utils import pairwise

name_list = [ 
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
        "top",
        "stomach"
]

COLOR_MAP = {
    'instance': (143, 35, 35),
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
    'thorax': (255, 255, 0),
    'pelvis': (255,0,170),
    'neck': (237,185,185),
    'top': (255, 0, 0),
    'stomach': (79, 35, 35)
}

EDGES_BY_NAME = [
    ['instance', 'neck'],
    ['neck', 'thorax'],
    ['thorax', 'left_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['thorax', 'right_shoulder'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['thorax', 'stomach'],
    ['stomach', 'pelvis'],
    ['pelvis', 'left_hip'],
    ['pelvis', 'right_hip'],
    ['left_hip', 'left_knee'],
    ['right_hip', 'right_knee'],
    ['left_knee', 'left_ankle'],
    ['right_knee', 'right_ankle'],
    ['instance', 'top'],
]

KEYPOINT_NAMES = ['instance'] + name_list 
EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'neck', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist']
TRACK_ORDER_1 = ['instance', 'neck', 'thorax', 'right_shoulder', 'right_elbow', 'right_wrist']
TRACK_ORDER_2 = ['instance', 'neck', 'thorax', 'stomach', 'pelvis', 'left_hip', 'left_knee', 'left_ankle']
TRACK_ORDER_3 = ['instance', 'neck', 'thorax', 'stomach', 'pelvis', 'right_hip', 'right_knee', 'right_ankle']
TRACK_ORDER_4 = ['instance', 'top']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4]

DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])

EPSILON = 1e-6


#print(len(DIRECTED_GRAPHS), DIRECTED_GRAPHS)
#print('edge:', len(EDGES))
#print('edge by name:', len(EDGES_BY_NAME))
