cifar100_fine_label = [
    'apple', # id 0
    'aquarium_fish', # id 1
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
    ]

cifar100_coarse_label = [
    'aquatic_mammals',
    'fish',
    'flowers',
    'food_containers',
    'fruit_and_vegetables',
    'household_electrical_devices',
    'household_furniture',
    'insects', #7
    'large_carnivores',
    'large_man-made_outdoor_things',
    'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores',
    'medium_mammals',
    'non-insect_invertebrates',
    'people',
    'reptiles',
    'small_mammals',
    'trees',
    'vehicles_1',
    'vehicles_2'
    ]

cifar100_mapping = {
'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'], #7
'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'people': ['baby', 'boy', 'girl', 'man', 'woman'],
'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


cifar100_valid_mapping = {
    "Bee": {
        "fine_id": 6,
        "fine_name": "bee",
        "coarse_id": 7,
        "coarse_name": "insects"
    },
    "Beetle": {
        "fine_id": 7,
        "fine_name": "beetle",
        "coarse_id": 7,
        "coarse_name": "insects"
    },
    "Butterfly": {
        "fine_id": 14,
        "fine_name": "butterfly",
        "coarse_id": 7,
        "coarse_name": "insects"
    },
    "Spider": {
        "fine_id": 79,
        "fine_name": "spider",
        "coarse_id": 13,
        "coarse_name": "non-insect_invertebrates"
    }

    # Ant, Dragonfly, Fly, Grasshopper, Ladybug, Mosquito, Wasp are not present in CIFAR-100
}

imagenet1k_valid_mapping = {
    "Ant": {
        310: "ant, emmet, pismire"
    },

    "Bee": {
        309: "bee"
    },

    "Beetle": {
        300: "tiger beetle",
        301: "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
        302: "ground beetle, carabid beetle",
        303: "long-horned beetle, longicorn, longicorn beetle",
        304: "leaf beetle, chrysomelid",
        305: "dung beetle",
        306: "rhinoceros beetle",
        307: "weevil"
    },

    "Butterfly": {
        321: "admiral butterfly",
        322: "ringlet butterfly",
        323: "monarch butterfly",
        324: "cabbage butterfly",
        325: "sulphur butterfly, sulfur butterfly",
        326: "lycaenid butterfly"
    },

    "Dragonfly": {
        319: "dragonfly",
        320: "damselfly"
    },

    "Fly": {
        308: "fly"
    },

    "Grasshopper": {
        311: "grasshopper, hopper"
    },

    "Ladybug": {
        301: "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle"
    },

    "Spider": {
        72: "black and gold garden spider",
        73: "barn spider",
        74: "garden spider",
        75: "black widow",
        76: "tarantula",
        77: "wolf spider"
    }

    # Mosquito and Wasp are not present in ImageNet-1k
}
