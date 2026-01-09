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

cifar100_to_clean_map = {
    6: "Bee",
    7: "Beetle",
    14: "Butterfly",
    79: "Spider"
    # Ant, Dragonfly, Fly, Grasshopper, Ladybug, Mosquito, Wasp are not present in CIFAR-100
}

iNat_to_clean_map = {
    # Ants
    "Camponotus planatus": "Ant",
    
    # Bees
    "Agapostemon virescens": "Bee",

    # Beetles
    "Alaus lusciosus": "Beetle",
    "Buprestis aurulenta": "Beetle",
    "Chrysolina americana": "Beetle",
    "Cicindela hirticollis": "Beetle",
    "Epicauta pennsylvanica": "Beetle",
    "Polydrusus formosus": "Beetle",
    "Trypoxylus dichotomus": "Beetle",

    # Butterflies
    "Aglais urticae": "Butterfly",
    "Anartia amathea": "Butterfly",
    "Cigaritis lohita": "Butterfly",
    "Euphyes vestris": "Butterfly",
    "Icaricia acmon": "Butterfly",
    "Icaricia lupini": "Butterfly",
    "Panoquina ocola": "Butterfly",

    # # Moths 
    # "Agrotis segetum": "Butterfly",
    # "Aphomia sociella": "Butterfly",
    # "Apogeshna stenialis": "Butterfly",
    # "Cabera pusaria": "Butterfly",
    # "Ceratomia catalpae": "Butterfly",
    # "Enyo lugubris": "Butterfly",
    # "Gastrina cristaria": "Butterfly",
    # "Gymnandrosoma punctidiscanum": "Butterfly",
    # "Heterophleps triguttaria": "Butterfly",
    # "Ochropleura plecta": "Butterfly",
    # "Thaumetopoea processionea": "Butterfly",
    # "Triphosa haesitata": "Butterfly",

    # Dragonflies
    
    # Flies

    # Grasshoppers
    "Gomphocerippus rufus": "Grasshopper",

    # Ladybugs
    "Hippodamia variegata": "Ladybug",

    # Mosquito

    # Spiders
    "Aphonopelma chalcodes": "Spider",
    "Platycryptus undatus": "Spider",

    # Wasps
    "Chlorion aerarium": "Wasp" #this wasp is blue though
}



imagenet1k_to_clean_map = {
    # Ant
    310: "Ant",

    # Bee
    309: "Bee",

    # Beetle
    300: "Beetle",
    302: "Beetle",
    303: "Beetle",
    304: "Beetle",
    305: "Beetle",
    306: "Beetle",
    307: "Beetle",

    # Ladybug
    301: "Ladybug",

    # Butterfly
    321: "Butterfly",
    322: "Butterfly",
    323: "Butterfly",
    324: "Butterfly",
    325: "Butterfly",
    326: "Butterfly",

    # Dragonfly
    319: "Dragonfly",
    320: "Dragonfly",

    # Fly
    308: "Fly",

    # Grasshopper
    311: "Grasshopper",

    # Spider
    72: "Spider",
    73: "Spider",
    74: "Spider",
    75: "Spider",
    76: "Spider",
    77: "Spider"

    # Mosquito and Wasp are not present in ImageNet-1k
}

# imagenet1k_valid_mapping = {
#     "Ant": {
#         310: "ant, emmet, pismire"
#     },

#     "Bee": {
#         309: "bee"
#     },

#     "Beetle": {
#         300: "tiger beetle",
#         301: "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
#         302: "ground beetle, carabid beetle",
#         303: "long-horned beetle, longicorn, longicorn beetle",
#         304: "leaf beetle, chrysomelid",
#         305: "dung beetle",
#         306: "rhinoceros beetle",
#         307: "weevil"
#     },

#     "Butterfly": {
#         321: "admiral butterfly",
#         322: "ringlet butterfly",
#         323: "monarch butterfly",
#         324: "cabbage butterfly",
#         325: "sulphur butterfly, sulfur butterfly",
#         326: "lycaenid butterfly"
#     },

#     "Dragonfly": {
#         319: "dragonfly",
#         320: "damselfly"
#     },

#     "Fly": {
#         308: "fly"
#     },

#     "Grasshopper": {
#         311: "grasshopper, hopper"
#     },

#     "Ladybug": {
#         301: "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle"
#     },

#     "Spider": {
#         72: "black and gold garden spider",
#         73: "barn spider",
#         74: "garden spider",
#         75: "black widow",
#         76: "tarantula",
#         77: "wolf spider"
#     }

#     # Mosquito and Wasp are not present in ImageNet-1k
# }
