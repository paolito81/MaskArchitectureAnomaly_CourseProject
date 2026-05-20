import torch

IGNORE_INDEX = 255

COCO_ID_TO_LABEL = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
    "80": "banner",
    "81": "blanket",
    "82": "bridge",
    "83": "cardboard",
    "84": "counter",
    "85": "curtain",
    "86": "door-stuff",
    "87": "floor-wood",
    "88": "flower",
    "89": "fruit",
    "90": "gravel",
    "91": "house",
    "92": "light",
    "93": "mirror-stuff",
    "94": "net",
    "95": "pillow",
    "96": "platform",
    "97": "playingfield",
    "98": "railroad",
    "99": "river",
    "100": "road",
    "101": "roof",
    "102": "sand",
    "103": "sea",
    "104": "shelf",
    "105": "snow",
    "106": "stairs",
    "107": "tent",
    "108": "towel",
    "109": "wall-brick",
    "110": "wall-stone",
    "111": "wall-tile",
    "112": "wall-wood",
    "113": "water-other",
    "114": "window-blind",
    "115": "window-other",
    "116": "tree-merged",
    "117": "fence-merged",
    "118": "ceiling-merged",
    "119": "sky-other-merged",
    "120": "cabinet-merged",
    "121": "table-merged",
    "122": "floor-other-merged",
    "123": "pavement-merged",
    "124": "mountain-merged",
    "125": "grass-merged",
    "126": "dirt-merged",
    "127": "paper-merged",
    "128": "food-other-merged",
    "129": "building-other-merged",
    "130": "rock-merged",
    "131": "wall-other-merged",
    "132": "rug-merged",
}

COCO_LABEL_TO_ID = {label: int(id) for id, label in COCO_ID_TO_LABEL.items()}

CITYSCAPES_ID_TO_LABEL = {
    "0": "road",
    "1": "sidewalk",
    "2": "building",
    "3": "wall",
    "4": "fence",
    "5": "pole",
    "6": "traffic light",
    "7": "traffic sign",
    "8": "vegetation",
    "9": "terrain",
    "10": "sky",
    "11": "person",
    "12": "rider",
    "13": "car",
    "14": "truck",
    "15": "bus",
    "16": "train",
    "17": "motorcycle",
    "18": "bicycle",
}

CITYSCAPES_LABEL_TO_ID = {
    label: int(id) for id, label in CITYSCAPES_ID_TO_LABEL.items()
}

CITYSCAPES_CLASSES = [
    CITYSCAPES_ID_TO_LABEL[str(i)] for i in range(len(CITYSCAPES_ID_TO_LABEL))
]


SHARED_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light",
]

SHARED_NAME_TO_ID = {label: idx for idx, label in enumerate(SHARED_CLASSES)}

CITYSCAPES_TO_SHARED = {
    CITYSCAPES_LABEL_TO_ID["person"]: SHARED_NAME_TO_ID["person"],
    CITYSCAPES_LABEL_TO_ID["car"]: SHARED_NAME_TO_ID["car"],
    CITYSCAPES_LABEL_TO_ID["truck"]: SHARED_NAME_TO_ID["truck"],
    CITYSCAPES_LABEL_TO_ID["bus"]: SHARED_NAME_TO_ID["bus"],
    CITYSCAPES_LABEL_TO_ID["motorcycle"]: SHARED_NAME_TO_ID["motorcycle"],
    CITYSCAPES_LABEL_TO_ID["bicycle"]: SHARED_NAME_TO_ID["bicycle"],
    CITYSCAPES_LABEL_TO_ID["traffic light"]: SHARED_NAME_TO_ID["traffic light"],
}

COCO_TO_SHARED = {
    COCO_LABEL_TO_ID["person"]: SHARED_NAME_TO_ID["person"],
    COCO_LABEL_TO_ID["car"]: SHARED_NAME_TO_ID["car"],
    COCO_LABEL_TO_ID["truck"]: SHARED_NAME_TO_ID["truck"],
    COCO_LABEL_TO_ID["bus"]: SHARED_NAME_TO_ID["bus"],
    COCO_LABEL_TO_ID["motorcycle"]: SHARED_NAME_TO_ID["motorcycle"],
    COCO_LABEL_TO_ID["bicycle"]: SHARED_NAME_TO_ID["bicycle"],
    COCO_LABEL_TO_ID["traffic light"]: SHARED_NAME_TO_ID["traffic light"],
}

CITYSCAPES_TO_CITYSCAPES = {
    cityscapes_id: cityscapes_id for cityscapes_id in CITYSCAPES_LABEL_TO_ID.values()
}

# Approximate mapping from COCO panoptic labels into the Cityscapes semantic
# label space. Classes that do not have a reasonable COCO counterpart are left
# unmapped and should be ignored during evaluation.
COCO_TO_CITYSCAPES = {
    COCO_LABEL_TO_ID["road"]: CITYSCAPES_LABEL_TO_ID["road"],
    COCO_LABEL_TO_ID["pavement-merged"]: CITYSCAPES_LABEL_TO_ID["sidewalk"],
    COCO_LABEL_TO_ID["building-other-merged"]: CITYSCAPES_LABEL_TO_ID["building"],
    COCO_LABEL_TO_ID["house"]: CITYSCAPES_LABEL_TO_ID["building"],
    COCO_LABEL_TO_ID["wall-brick"]: CITYSCAPES_LABEL_TO_ID["wall"],
    COCO_LABEL_TO_ID["wall-stone"]: CITYSCAPES_LABEL_TO_ID["wall"],
    COCO_LABEL_TO_ID["wall-tile"]: CITYSCAPES_LABEL_TO_ID["wall"],
    COCO_LABEL_TO_ID["wall-wood"]: CITYSCAPES_LABEL_TO_ID["wall"],
    COCO_LABEL_TO_ID["wall-other-merged"]: CITYSCAPES_LABEL_TO_ID["wall"],
    COCO_LABEL_TO_ID["fence-merged"]: CITYSCAPES_LABEL_TO_ID["fence"],
    COCO_LABEL_TO_ID["traffic light"]: CITYSCAPES_LABEL_TO_ID["traffic light"],
    COCO_LABEL_TO_ID["stop sign"]: CITYSCAPES_LABEL_TO_ID["traffic sign"],
    COCO_LABEL_TO_ID["tree-merged"]: CITYSCAPES_LABEL_TO_ID["vegetation"],
    COCO_LABEL_TO_ID["grass-merged"]: CITYSCAPES_LABEL_TO_ID["terrain"],
    COCO_LABEL_TO_ID["dirt-merged"]: CITYSCAPES_LABEL_TO_ID["terrain"],
    COCO_LABEL_TO_ID["sky-other-merged"]: CITYSCAPES_LABEL_TO_ID["sky"],
    COCO_LABEL_TO_ID["person"]: CITYSCAPES_LABEL_TO_ID["person"],
    COCO_LABEL_TO_ID["car"]: CITYSCAPES_LABEL_TO_ID["car"],
    COCO_LABEL_TO_ID["truck"]: CITYSCAPES_LABEL_TO_ID["truck"],
    COCO_LABEL_TO_ID["bus"]: CITYSCAPES_LABEL_TO_ID["bus"],
    COCO_LABEL_TO_ID["train"]: CITYSCAPES_LABEL_TO_ID["train"],
    COCO_LABEL_TO_ID["motorcycle"]: CITYSCAPES_LABEL_TO_ID["motorcycle"],
    COCO_LABEL_TO_ID["bicycle"]: CITYSCAPES_LABEL_TO_ID["bicycle"],
}


def remap_target_ids(target, id_map, ignore_index=IGNORE_INDEX):
    remapped = target.new_full(target.shape, ignore_index)
    for src_id, dst_id in id_map.items():
        remapped[target == src_id] = dst_id
    return remapped


def remap_logits(logits, id_map, num_shared):
    shared = logits.new_zeros((num_shared, *logits.shape[1:]))
    for src_id, dst_id in id_map.items():
        shared[dst_id] += logits[src_id]
    return shared
