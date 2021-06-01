import sys
import h5py
import os
import json
import numpy as np

sys.path.append('../../')
from data.thor_constants import THOR_OBJECT_TYPES
from functools import lru_cache
from nltk.metrics.distance import edit_distance
from tqdm import tqdm
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm", disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])
nlp_lg = spacy.load("en_core_web_lg", disable=['textcat', 'tagger', 'parser', 'ner'])


def _vec_avg(x):
    return np.stack([y.vector for y in nlp_lg(x)]).mean(0)


# Get VICO object types
def get_vico(words_only=True):
    VICO_PATH = '/home/rowan/datasets3/vico/glove_300_vico_linear_100/'

    with open(os.path.join(VICO_PATH, 'visual_words.json'), 'r') as f:
        visual_words = json.load(f)
    with open(os.path.join(VICO_PATH, 'visual_word_vecs_idx.json'), 'r') as f:
        word_to_idx = json.load(f)

    visual_words_exc = [vw for vw in visual_words if vw not in word_to_idx]
    print("These words excluded: {}".format(','.join(visual_words_exc)))
    visual_words = sorted([vw for vw in visual_words if vw in word_to_idx],
                          key=lambda x: word_to_idx[x])
    if words_only:
        return visual_words

    with h5py.File(os.path.join(VICO_PATH, 'visual_word_vecs.h5py'), 'r') as fn:
        visual_embeddings = fn['embeddings'][np.array([word_to_idx[vw] for vw in visual_words])]
    return visual_words, visual_embeddings


vico_words = get_vico()


def get_coco():
    COCO_PATH = '/home/rowan/datasets3/mscoco/annotations/instances_val2014.json'
    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)
    coco_names = [x['name'] for x in coco['categories']]
    return coco_names


coco_categories = get_coco()

# Create joint category names

vg_categories = []
with open('/home/rowan/code/backbones/vgpretrain_data/genome/1600-400-20/objects_vocab.txt', 'r') as f:
    for i, l in enumerate(f):
        vg_categories.append(l.strip())

# Open images categories
openimages_categories = []
with open('/home/rowan/datasets3/openimages/class-descriptions-boxable.csv', 'r') as f:
    for l in f:
        openimages_categories.append(l.strip().split(',')[1])


@lru_cache(4096)
def simplify_word(word):
    if word.lower().startswith('human '):
        word = word[6:].strip()

    word = word.replace(' ', '').lower()
    word = {
        'doughnut': 'donut',
        'sandwhich': 'sandwich',
    }.get(word, word)

    # Simple plurals
    if word.endswith('s'):
        word_lemma = nlp(word)[0].lemma_

        # Stripping S is OK
        if word_lemma in (word[:-1], word):
            return word_lemma
        # else:
        #     print("WEIRD. {} vs {}".format(word_lemma, word[:-1]))
    return word


# Create joint categorie
@lru_cache(1024)
def _edit_distance_cached(x1, x2):
    return edit_distance(x1, x2)


def category_distance(x1, x2):
    if ',' in x1:
        return min(category_distance(y, x2) for y in x1.split(','))
    if ',' in x2:
        return min(category_distance(x1, y) for y in x2.split(','))
    x1_lemma = simplify_word(x1)
    x2_lemma = simplify_word(x2)

    dist = edit_distance(x1_lemma, x2_lemma) if x1_lemma > x2_lemma else edit_distance(x2_lemma, x1_lemma)
    return dist / max(len(x1), len(x2))


ALL_DATASETS = [('thor', THOR_OBJECT_TYPES), ('coco', coco_categories), ('vg', vg_categories),
                ('oi', openimages_categories)]

# First create a giant DF and then we'll merge
category_to_name = []
for dataset, dataset_objects in ALL_DATASETS:
    for obj in dataset_objects:
        item = {f'{ds[0]}_name': None for ds in ALL_DATASETS}
        item[f'{dataset}_name'] = obj
        category_to_name.append(item)


def _compute_distance_matrix(category_to_name):
    dist_mat = np.zeros((len(category_to_name), len(category_to_name)),
                        dtype=np.float32)

    names = [','.join([z for x in v1.values() if x is not None for z in x.split(',')])
             for v1 in category_to_name]

    for i, v1 in enumerate(tqdm(names)):
        for j, v2 in enumerate(names[(i + 1):]):
            dist_mat[i, i + j + 1] = category_distance(v1, v2)
    dist_mat += 2.0 * np.tril(np.ones(dist_mat.shape, dtype=np.float32))
    return dist_mat


dist_mat = _compute_distance_matrix(category_to_name)

def _merge(parent, child):
    for k in sorted(category_to_name[parent]):
        vs = []
        if category_to_name[parent][k] is not None:
            vs += category_to_name[parent][k].split(',')
        if category_to_name[child][k] is not None:
            vs += category_to_name[child][k].split(',')
        if len(vs) > 0:
            category_to_name[parent][k] = ','.join(vs)

    category_to_name = [x for i, x in enumerate(category_to_name) if i != child]
    idx = np.ones(dist_mat.shape[0], dtype=np.bool)
    idx[child] = False
    dist_mat = dist_mat[idx]
    dist_mat = dist_mat[:, idx]


# Find the MOST SIMILAR THINGS
for i in range(1000):
    mi = np.argmin(dist_mat)
    child = mi % dist_mat.shape[0]
    parent = mi // dist_mat.shape[0]

    if (category_to_name[parent] is None) or (category_to_name[child] is None):
        print("Uh oh recomputing")
        category_to_name = [x for x in category_to_name if x is not None]
        dist_mat = _compute_distance_matrix(category_to_name)

    if (category_to_name[child]['thor_name'] is not None) and (category_to_name[parent]['thor_name'] is not None):
        dist_mat[child, parent] = 9000.0
        dist_mat[parent, child] = 9000.0
        continue

    if dist_mat[parent, child] > 0.12:
        break
    print("Distance {:.3f}: Merging {} -> {}".format(dist_mat[parent, child],
                                                     category_to_name[child], category_to_name[parent]), flush=True)
    _merge(parent, child)


# Do this once again but with GLOVE distances
names = [' '.join([z for x in v1.values() if x is not None for z in x.split(',')])
         for v1 in category_to_name]

vec_reps = np.stack([_vec_avg(n) for n in tqdm(names)])
vec_reps = vec_reps / np.sqrt(np.square(vec_reps).sum(1) + 1e-8)[:, None]
sims = (vec_reps @ vec_reps.T)
dist_mat = 1.0 - sims
dist_mat += 100.0 * np.tril(np.ones(dist_mat.shape, dtype=np.float32))

# Handle manual merges - this took forever
manual_merges = {(('microwave', None, 'Microwave', 'microwave,microwave oven'),
  (None, 'Microwave oven', None, None)): True,
 ((None, None, None, 'pool'), (None, 'Swimming pool', None, None)): True,
 ((None, None, 'Fridge', 'refrigerator,fridge'),
  ('refrigerator', 'Refrigerator', None, None)): True,
 ((None, None, None, 'tree branches'),
  (None, None, None, 'tree branch')): True,
 ((None, None, None, 'police officer'), (None, None, None, 'police')): True,
 ((None, 'Shirt', None, 'shirt'), (None, None, None, 'tee shirt')): True,
 (('laptop', 'Laptop', 'Laptop', 'laptop,laptops'),
  (None, None, None, 'laptop computer')): True,
 ((None, None, None, 'phone'), (None, 'Mobile phone', None, None)): True,
 ((None, None, 'Sofa', 'sofa'), (None, 'Sofa bed', None, None)): True,
 ((None, None, 'Pan', 'pan,pans'), (None, 'Frying pan', None, None)): True,
 ((None, 'Wheel', None, 'wheel,wheels'),
  (None, None, None, 'steering wheel')): False,
 ((None, None, None, 'windshield'),
  (None, None, None, 'windshield wipers,windshield wiper')): False,
 ((None, None, None, 'police officer,police'),
  (None, None, None, 'officer')): True,
 ((None, None, None, 'bathroom'), (None, None, None, 'bathroom sink')): False,
 (('boat', 'Boat', None, 'boat,boats'),
  (None, None, None, 'sailboat,sail boat')): False,
 ((None, None, None, 'ski pants'), (None, None, None, 'ski jacket')): False,
 ((None, 'Dress', None, 'dress'), (None, None, None, 'dress shirt')): False,
 (('teddy bear', 'Teddy bear', 'TeddyBear', 'teddy bear,teddy bears'),
  (None, None, None, 'teddy')): True,
 ((None, 'Beer', None, 'beer'), (None, None, None, 'beer bottle')): False,
 ((None, None, None, 'sail'), (None, None, None, 'sailboat,sail boat')): False,
 ((None, None, 'WineBottle', 'wine bottle'),
  (None, 'Wine', None, 'wine')): False,
 ((None, None, None, 'front window'),
  (None, None, None, 'side window')): False,
 (('cell phone', None, 'CellPhone', 'cell phone,cellphone'),
  (None, 'Mobile phone', None, 'phone')): True,
 ((None, 'Taxi', None, 'taxi'), (None, None, None, 'taxi cab')): True,
 ((None, 'Bathtub', 'Bathtub', 'bathtub,bath tub'),
  (None, None, None, 'tub')): True,
 (('dining table', None, 'DiningTable', None),
  (None, 'Kitchen & dining room table', None, None)): True,
 ((None, 'Necklace', None, 'necklace'),
  (None, 'Earrings', None, 'earrings,earring')): False,
 ((None, None, 'Floor', 'floor,floors'),
  (None, None, None, 'tile floor')): False,
 ((None, None, None, 'garage door'), (None, None, None, 'garage')): False,
 ((None, None, None, 'garage door'),
  (None, 'Door', None, 'door,doors')): False,
 ((None, 'Glasses', None, 'glasses'),
  (None, None, None, 'eyeglasses,eye glasses')): True,
 ((None, 'Shirt', None, 'shirt,tee shirt'),
  (None, None, None, 'dress shirt')): False,
 (('bottle', 'Bottle', 'Bottle', 'bottle,bottles'),
  (None, None, None, 'beer bottle')): False,
 ((None, 'Tomato', 'Tomato', 'tomato'), (None, None, None, 'tomatoes')): True,
 ((None, None, None, 'stone wall'), (None, None, None, 'brick wall')): False,
 (('tennis racket',
   'Tennis racket',
   'TennisRacket',
   'tennis racket,tennis racquet'),
  (None, 'Table tennis racket', None, None)): False,
 (('cake', 'Cake', None, 'cake'), (None, None, None, 'birthday cake')): True,
 ((None, None, None, 'train station'), (None, None, None, 'station')): False,
 (('tennis racket',
   'Tennis racket',
   'TennisRacket',
   'tennis racket,tennis racquet'),
  (None, None, None, 'tennis')): False,
 ((None, None, None, 'wall,walls'), (None, None, None, 'brick wall')): False,
 ((None, None, None, 'cardboard box'), (None, None, None, 'cardboard')): False,
 ((None, None, None, 'back wheel'), (None, None, None, 'front wheel')): False,
 ((None, None, None, 'tennis player'),
  (None, None, None, 'soccer player')): False,
 ((None, 'Tree', None, 'tree,trees'),
  (None, None, None, 'pine tree,pine trees')): False,
 ((None, None, None, 'soccer ball'),
  (None, 'Tennis ball', None, 'tennis ball')): False,
 ((None, None, None, 'hoodie'), (None, None, None, 'sweatshirt')): True,
 ((None, None, None, 'ocean'), (None, None, None, 'ocean water')): True,
 ((None, None, None, 'mask'), (None, None, None, 'face mask')): True,
 ((None, 'Girl', None, 'girl,girls'), (None, None, None, 'little girl')): True,
 ((None, 'Tree', None, 'tree,trees'),
  (None, None, None, 'tree branches,tree branch')): False,
 ((None, 'Cello', None, None), (None, 'Violin', None, None)): False,
 ((None, None, None, 'baseball player,baseball players'),
  (None, None, None, 'baseball game')): False,
 ((None, 'Cream', None, 'cream'), (None, 'Ice cream', None, None)): False,
 ((None, 'Jacket', None, 'jacket'), (None, None, None, 'suit jacket')): False,
 (('teddy bear', 'Teddy bear', 'TeddyBear', 'teddy bear,teddy bears,teddy'),
  ('bear', 'Bear', None, 'bear,bears')): False,
 (('bus', 'Bus', None, 'bus'), (None, None, None, 'bus stop')): False,
 ((None, None, 'WineBottle', 'wine bottle'),
  (None, None, None, 'beer bottle')): False,
 (('keyboard', None, None, 'keyboard'),
  (None, 'Computer keyboard', None, 'computer keyboard')): True,
 ((None, None, 'ShowerCurtain', 'shower curtain'),
  (None, 'Shower', None, 'shower')): False,
 ((None, None, 'WineBottle', 'wine bottle'),
  ('wine glass', 'Wine glass', None, 'wine glass')): False,
 (('cup', None, 'Cup', 'cup'),
  (None, 'Coffee cup', None, 'coffee cup')): False,
 ((None, None, None, 'clothes'), (None, 'Clothing', None, 'clothing')): True,
 ((None, 'Sculpture', None, 'sculpture'),
  (None, 'Bronze sculpture', None, None)): False,
 ((None, 'Wheel', None, 'wheel,wheels'),
  (None, None, None, 'front wheel')): False,
 ((None, None, None, 'bow tie'), (None, None, None, 'bow')): False,
 ((None, 'Tennis ball', None, 'tennis ball'),
  (None, 'Golf ball', None, None)): False,
 (('sports ball', None, None, None), (None, None, None, 'soccer ball')): False,
 (('wine glass', 'Wine glass', None, 'wine glass'),
  (None, None, None, 'wine glasses')): True,
 (('potted plant', None, None, None),
  (None, 'Plant', None, 'plant,plants')): False,
 (('sink', 'Sink', 'Sink', 'sink,sinks'),
  (None, None, None, 'bathroom sink')): False,
 (('broccoli', 'Broccoli', None, 'broccoli'),
  (None, None, None, 'cauliflower')): False,
 ((None, None, None, 'soccer ball'), (None, 'Rugby ball', None, None)): False,
 ((None, 'Man', None, 'man'), (None, None, None, 'young man')): False,
 ((None, None, None, 'living room'), (None, None, None, 'room')): False,
 ((None, None, None, 'plastic'), (None, 'Plastic bag', None, None)): False,
 ((None, 'Alarm clock', 'AlarmClock', 'alarm clock'),
  ('clock', 'Clock', None, 'clock,clocks')): False,
 ((None, None, None, 'branches'),
  (None, None, None, 'tree branches,tree branch')): True,
 ((None, 'Potato', 'Potato', 'potato'), (None, None, None, 'potatoes')): True,
 ((None, None, None, 'wall,walls'), (None, None, None, 'stone wall')): False,
 ((None, None, None, 'sea'), (None, 'Sea turtle', None, None)): False,
 ((None, None, None, 'dirt'), (None, None, None, 'dirt road')): False,
 ((None, None, None, 'tail feathers'),
  (None, None, None, 'feathers,feather')): False,
 ((None, None, None, 'frame'),
  (None, 'Picture frame', None, 'picture frame')): False,
 ((None, 'Saxophone', None, None), (None, 'Trombone', None, None)): False,
 ((None, None, None, 'life jacket'), (None, None, None, 'life vest')): True,
 ((None, None, None, 'ankle'), (None, None, None, 'knee')): False,
 ((None, None, None, 'kitchen'), (None, 'Kitchen utensil', None, None)): False,
 (('pizza', 'Pizza', None, 'pizza,pizzas'),
  (None, None, None, 'pizza slice')): False,
 ((None, 'Necklace', None, 'necklace'),
  (None, None, None, 'bracelet,bracelets')): False,
 (('broccoli', 'Broccoli', None, 'broccoli'),
  (None, None, None, 'spinach')): False,
 (('wine glass', 'Wine glass', None, 'wine glass,wine glasses'),
  (None, 'Wine', None, 'wine')): False,
 (('bicycle', 'Bicycle', None, 'bicycle,bicycles'),
  (None, None, None, 'bike,bikes')): True,
 (('car', 'Car', None, 'car,cars'),
  (None, None, None, 'passenger car')): False,
 ((None, None, None, 'pepper,peppers'),
  (None, 'Bell pepper', None, None)): True,
 (('refrigerator', 'Refrigerator', 'Fridge', 'refrigerator,fridge'),
  (None, None, None, 'freezer')): False,
 ((None, None, None, 'exhaust pipe'), (None, None, None, 'pipe,pipes')): False,
 ((None, None, None, 'glass'), (None, None, None, 'glass door')): False,
 ((None, None, None, 'baseball player,baseball players'),
  (None, None, None, 'baseball')): False,
 ((None, None, 'TomatoSliced', 'tomato slice'),
  (None, None, None, 'slice,slices')): False,
 ((None, None, None, 'tiles,tile'), (None, None, None, 'tile floor')): False,
 ((None, None, None, 'luggage'), (None, 'Luggage and bags', None, None)): True,
 ((None, None, None, 'wipers'),
  (None, None, None, 'windshield wipers,windshield wiper')): True,
 ((None, None, None, 'water'), (None, None, None, 'water tank')): False,
 ((None, None, None, 'tail fin'), (None, None, None, 'fin')): False,
 ((None, None, 'Pencil', 'pencil'),
  (None, 'Pencil sharpener', None, None)): False,
 ((None, 'Tank', None, 'tank'), (None, None, None, 'water tank')): False,
 ((None, 'Wheel', None, 'wheel,wheels'),
  (None, 'Bicycle wheel', None, None)): False,
 (('baseball bat', 'Baseball bat', 'BaseballBat', 'baseball bat'),
  (None, None, None, 'baseball')): False,
 ((None, None, None, 'tail feathers'),
  (None, None, None, 'tail,tails')): False,
 ((None, None, None, 'dinner'), (None, None, None, 'lunch')): False,
 ((None, None, None, 'stove'), (None, 'Gas stove', None, None)): False,
 ((None, None, None, 'stairway'), (None, None, None, 'staircase')): True,
 (('cake', 'Cake', None, 'cake,birthday cake'),
  (None, 'Cake stand', None, None)): False,
 ((None, None, None, 'tail wing'), (None, None, None, 'wing,wings')): False,
 ((None, None, None, 'soccer player'),
  (None, None, None, 'baseball player,baseball players')): False,
 ((None, None, None, 'bin'), (None, None, None, 'trash bin')): False,
 (('banana', 'Banana', None, 'bananas,banana'),
  (None, None, None, 'banana slice')): False,
 ((None, None, None, 'game'), (None, None, None, 'baseball game')): False,
 (('sports ball', None, None, None),
  (None, 'Tennis ball', None, 'tennis ball')): False,
 (('chair', 'Chair', 'Chair', 'chair,chairs'),
  (None, None, None, 'lounge chair')): True,
 ((None, None, None, 'stove'),
  (None, None, None, 'stove top,stovetop')): False,
 ((None, 'Palm tree', None, 'palm trees,palm tree'),
  (None, None, None, 'palm')): False,
 ((None, None, None, 'bag,bags'), (None, 'Plastic bag', None, None)): False,
 ((None, 'Lavender (Plant)', None, None),
  (None, 'Squash (Plant)', None, None)): False,
 ((None, None, None, 'frosting'), (None, None, None, 'icing')): True,
 (('train', 'Train', None, 'train,trains'),
  (None, None, None, 'train station')): False,
 ((None, None, 'TVStand', 'tv stand'), ('tv', None, None, None)): False,
 ((None, None, None, 'trash can,trashcan'),
  (None, None, None, 'trash')): False,
 ((None, None, None, 'blueberries'),
  (None, None, None, 'strawberries')): False,
 ((None, None, None, 'light,lights'), (None, 'Light bulb', None, None)): False,
 ((None, None, None, 'clock face'), (None, None, None, 'clock hand')): False,
 ((None, None, None, 'banana peel'), (None, None, None, 'peel')): False,
 (('chair', 'Chair', 'Chair', 'chair,chairs,lounge chair'),
  (None, None, None, 'office chair')): False,
 ((None, 'Toilet paper', 'ToiletPaper', 'toilet paper'),
  (None, 'Paper towel', None, 'paper towels,paper towel')): False,
 ((None, None, None, 'player,players'),
  (None, None, None, 'soccer player')): False,
 ((None, None, None, 'baseball game'), (None, None, None, 'baseball')): False,
 ((None, 'Bell pepper', None, 'pepper,peppers'),
  (None, None, None, 'onions,onion')): False,
 ((None, None, None, 'tennis match'), (None, None, None, 'tennis')): False,
 ((None, None, None, 'screen'), (None, None, None, 'computer screen')): False,
 ((None, None, None, 'ski pants'), (None, None, None, 'ski suit')): False,
 ((None, 'Tree', None, 'tree,trees'),
  (None, 'Palm tree', None, 'palm trees,palm tree')): False,
 ((None, None, None, 'front wheel'),
  (None, None, None, 'steering wheel')): False,
 ((None, None, 'ShowerDoor', 'shower door'),
  (None, 'Shower', None, 'shower')): False,
 ((None, None, None, 'baseball field'),
  (None, None, None, 'soccer field')): False,
 ((None, None, None, 'signal'), (None, None, None, 'signal light')): False,
 ((None, 'Cricket ball', None, None), (None, 'Rugby ball', None, None)): False,
 ((None, None, None, 'tank top'), (None, 'Tank', None, 'tank')): False,
 ((None, None, None, 'headlight,headlights,head light'),
  (None, None, None, 'tail light,taillight,tail lights')): False,
 ((None, None, 'Cabinet', 'cabinets,cabinet'),
  (None, None, None, 'cabinet door')): False,
 ((None, None, 'Cabinet', 'cabinets,cabinet'),
  (None, 'Bathroom cabinet', None, None)): False,
 ((None, 'Tea', None, 'tea'), (None, None, None, 'tea kettle')): False,
 ((None, None, None, 'dirt bike'), (None, None, None, 'dirt road')): False,
 ((None, None, None, 'kitchen'),
  (None, 'Kitchen appliance', None, None)): False,
 ((None, None, None, 'office'), (None, 'Office building', None, None)): False,
 ((None, None, None, 'machine'), (None, 'Washing machine', None, None)): False,
 ((None, 'Helmet', None, 'helmet'),
  (None, 'Bicycle helmet', None, None)): False,
 ((None, 'Window', 'Window', 'window,windows'),
  (None, None, None, 'front window')): False,
 ((None, 'Coffee', None, 'coffee'),
  (None, 'Coffee cup', None, 'coffee cup')): False,
 ((None, 'French fries', None, 'french fries'),
  (None, None, None, 'fries')): True,
 ((None, None, None, 'water'), (None, None, None, 'water bottle')): False,
 ((None, None, None, 'bathroom'),
  (None, 'Bathroom cabinet', None, None)): False,
 ((None, 'Coffee', None, 'coffee'), (None, None, None, 'coffee pot')): False,
 (('tie', 'Tie', None, 'tie'), (None, None, None, 'bow tie')): False,
 (('baseball glove', 'Baseball glove', None, 'baseball glove'),
  (None, None, None, 'baseball')): False,
 ((None, None, None, 'front wheel'), (None, None, None, 'front')): False,
 ((None, 'Door', None, 'door,doors'), (None, None, None, 'glass door')): False,
 ((None, None, None, 'grass'), (None, None, None, 'grass field')): False,
 (('dining table', 'Kitchen & dining room table', 'DiningTable', None),
  (None, 'Table', None, 'table,tables')): False,
 ((None, None, None, 'pitcher'), (None, None, None, "pitcher's mound")): False,
 (('knife', 'Knife', 'Knife', 'knife'), (None, None, None, 'knives')): True,
 ((None, 'Box', 'Box', 'box'), (None, None, None, 'cardboard box')): False,
 ((None, 'Coffee table', 'CoffeeTable', 'coffee table'),
  (None, 'Coffee', None, 'coffee')): False,
 ((None, None, None, 'baseball uniform'),
  (None, 'Sports uniform', None, None)): False,
 ((None, None, None, 'passenger,passengers'),
  (None, None, None, 'passenger train')): False,
 ((None, 'Tennis ball', None, 'tennis ball'),
  (None, None, None, 'tennis')): False,
 ((None, None, None, 'plumbing'),
  (None, 'Plumbing fixture', None, None)): False,
 ((None, None, None, 'machine'), (None, 'Sewing machine', None, None)): False,
 (('wine glass', 'Wine glass', None, 'wine glass,wine glasses'),
  (None, None, None, 'glass')): False,
 ((None, None, None, 'soda'), (None, None, None, 'soda can')): False,
 ((None, None, 'SaltShaker', 'salt shaker'),
  (None, 'Salt and pepper shakers', None, None)): True,
 ((None, None, None, 'baseball field'), (None, None, None, 'baseball')): False,
 ((None, None, None, 'soap'),
  (None, 'Soap dispenser', None, 'soap dispenser')): False,
 ((None, None, None, 'photos,photo'),
  (None, None, None, 'picture,pictures')): True,
 (('tennis racket',
   'Tennis racket',
   'TennisRacket',
   'tennis racket,tennis racquet'),
  (None, 'Racket', None, 'racket,racquet')): True,
 ((None, None, None, 'road'), (None, None, None, 'dirt road')): False,
 ((None, None, None, 'dirt bike'), (None, None, None, 'dirt')): False,
 ((None, 'Door', None, 'door,doors'),
  (None, None, None, 'cabinet door')): False,
 ((None, None, None, 'surf'), (None, None, None, 'surfing')): False,
 (('bottle', 'Bottle', 'Bottle', 'bottle,bottles'),
  (None, None, None, 'water bottle')): False,
 (('bicycle', 'Bicycle', None, 'bicycle,bicycles,bike,bikes'),
  (None, 'Stationary bicycle', None, None)): False,
 ((None, None, None, 'hill,hills'),
  (None, None, None, 'hill side,hillside')): False,
 (('banana', 'Banana', None, 'bananas,banana'),
  (None, None, None, 'banana peel')): False,
 (('mouse', 'Mouse', None, 'mouse'),
  (None, 'Computer mouse', None, 'computer mouse')): False,
 ((None, None, None, 'monitor,monitors'),
  (None, 'Computer monitor', None, 'computer monitor')): True,
 (('clock', 'Clock', None, 'clock,clocks'),
  (None, None, None, 'clock hand')): False,
 ((None, None, None, 'street lamp'),
  (None, 'Street light', None, 'streetlight,street light')): True,
 ((None, 'Traffic sign', None, 'traffic sign'),
  (None, None, None, 'traffic')): False,
 ((None, None, None, 'air'), (None, None, None, 'air vent')): False,
 ((None, 'Remote control', 'RemoteControl', 'remote control'),
  ('remote', None, None, 'remotes,remote')): True,
 ((None, None, None, 'player,players'),
  (None, None, None, 'baseball player,baseball players')): False,
 ((None, 'Shirt', None, 'shirt,tee shirt'),
  (None, None, None, 'tshirt,t-shirt,t shirt')): True,
 ((None, None, None, 'tablet'), (None, 'Tablet computer', None, None)): True,
 ((None, 'Cocktail', None, None),
  (None, 'Cocktail shaker', None, None)): False,
 (('bicycle', 'Bicycle', None, 'bicycle,bicycles,bike,bikes'),
  (None, 'Bicycle wheel', None, None)): False,
 ((None, None, None, 'store'), (None, 'Convenience store', None, None)): False,
 ((None, 'Television', 'Television', 'television,tv'),
  ('tv', None, None, None)): True,
 ((None, 'Food', None, 'food'), (None, 'Fast food', None, None)): False,
 ((None, None, None, 'stuffed animals,stuffed animal'),
  (None, None, None, 'stuffed bear')): False,
 (('bear', 'Bear', None, 'bear,bears'),
  (None, 'Polar bear', None, 'polar bear')): False,
 ((None, None, None, 'fence'), (None, None, None, 'wire fence')): False,
 ((None, None, None, 'name'), (None, None, None, 'name tag')): False,
 (('oven', 'Oven', None, 'oven'), (None, None, None, 'oven door')): False,
 ((None, None, None, 'front window'), (None, None, None, 'front')): False,
 ((None, 'Sea turtle', None, None), (None, 'Turtle', None, None)): False,
 ((None, 'French fries', None, 'french fries,fries'),
  (None, None, None, 'french fry')): True,
 ((None, None, None, 'ski boot,ski boots'),
  (None, None, None, 'ski pants')): False,
 ((None, 'Container', None, 'container,containers'),
  (None, 'Waste container', None, None)): False,
 ((None, None, None, 'bag,bags'), (None, None, None, 'trash bag')): False,
 ((None, None, None, 'bathroom sink'),
  (None, 'Bathroom cabinet', None, None)): False,
 (('skis', 'Ski', None, 'skis,ski'),
  (None, None, None, 'ski pole,ski poles')): False,
 ((None, 'Hat', None, 'hat'), (None, 'Cowboy hat', None, 'cowboy hat')): False,
 ((None, None, None, 'ski boot,ski boots'),
  (None, None, None, 'ski jacket')): False,
 (('car', 'Car', None, 'car,cars'),
  (None, None, None, 'train car,train cars')): False,
 ((None, None, 'ShowerHead', 'shower head'),
  (None, 'Shower', None, 'shower')): False,
 ((None, 'Human hair', None, 'hair'),
  (None, None, None, 'facial hair')): False,
 ((None, None, None, 'rack'), (None, None, None, 'towel rack')): False,
 ((None, None, None, 'vent'), (None, None, None, 'air vent')): False,
 (('baseball bat', 'Baseball bat', 'BaseballBat', 'baseball bat'),
  (None, None, None, 'bats,bat')): False,
 ((None, 'Salt and pepper shakers', 'SaltShaker', 'salt shaker'),
  (None, None, None, 'salt')): False,
 ((None, 'Wheel', None, 'wheel,wheels'),
  (None, None, None, 'back wheel')): False,
 ((None, 'Pillow', 'Pillow', 'pillow,pillows'),
  (None, None, None, 'throw pillow')): False,
 ((None, 'Tank', None, 'tank'), (None, None, None, 'gas tank')): False,
 ((None, None, None, 'tennis shoe,tennis shoes'),
  (None, None, None, 'tennis')): False,
 (('dog', 'Dog', None, 'dog,dogs'), (None, None, None, 'puppy')): False,
 (('train', 'Train', None, 'train,trains'),
  (None, None, None, 'passenger train')): False,
 ((None, 'Mammal', None, None), (None, 'Marine mammal', None, None)): False,
 ((None, None, None, 'onions,onion'), (None, None, None, 'celery')): False,
 (('banana', 'Banana', None, 'bananas,banana'),
  (None, None, None, 'banana bunch')): True,
 ((None, None, None, 'baseball'),
  (None, None, None, 'baseball uniform')): False,
 (('bowl', 'Bowl', 'Bowl', 'bowl,bowls'),
  (None, 'Mixing bowl', None, None)): False,
 ((None, 'Tree', None, 'tree,trees'), (None, 'Tree house', None, None)): False,
 ((None, 'Lamp', None, 'lamp,lamps'), (None, None, None, 'table lamp')): False,
 ((None, None, None, 'brick wall'), (None, None, None, 'bricks,brick')): False,
 ((None, 'Kitchen utensil', None, None),
  (None, 'Kitchen knife', None, None)): False,
 ((None, None, None, 'utensils,utensil'),
  (None, 'Kitchen utensil', None, None)): True,
 ((None, None, None, 'ski suit'), (None, None, None, 'ski jacket')): False,
 (('bowl', 'Bowl', 'Bowl', 'bowl,bowls'),
  (None, None, None, 'toilet bowl')): False,
 ((None, None, None, 'numerals,numeral'),
  (None, None, None, 'roman numerals,roman numeral')): False,
 ((None, None, None, 'goatee'),
  (None, None, None, 'moustache,mustache')): False,
 ((None, None, None, 'engine,engines'),
  (None, None, None, 'jet engine')): False,
 ((None, None, 'TomatoSliced', 'tomato slice'),
  (None, None, None, 'banana slice')): False,
 ((None, 'Wine', None, 'wine'), (None, 'Wine rack', None, None)): False,
 ((None, None, None, 'tennis player'), (None, None, None, 'tennis')): False,
 ((None, None, None, 'office chair'), (None, None, None, 'office')): False,
 ((None, 'Window', 'Window', 'window,windows'),
  (None, None, None, 'side window')): False,
 ((None, None, 'GarbageCan', 'garbage can'),
  (None, None, None, 'trash can,trashcan')): True,
 (('baseball bat', 'Baseball bat', 'BaseballBat', 'baseball bat'),
  ('baseball glove', 'Baseball glove', None, 'baseball glove')): False,
 (('sports ball', None, None, None), (None, 'Golf ball', None, None)): False,
 ((None, None, None, 'ski pants'), (None, None, None, 'snow pants')): True,
 ((None, 'Kettle', 'Kettle', 'kettle'),
  (None, None, None, 'tea kettle')): True,
 ((None, 'Towel', 'Towel', 'towel,towels'),
  (None, None, None, 'towel rack')): False,
 ((None, 'Mirror', 'Mirror', 'mirror'),
  (None, None, None, 'side mirror')): False,
 ((None, None, None, 'dishes'), (None, None, None, 'dish')): True,
 ((None, None, None, 'dispenser'),
  (None, 'Soap dispenser', None, 'soap dispenser')): False,
 ((None, None, None, 'front legs'), (None, None, None, 'front')): False,
 ((None, 'Building', None, 'building,buildings'),
  (None, 'Office building', None, None)): False,
 (('remote',
   'Remote control',
   'RemoteControl',
   'remote control,remotes,remote'),
  (None, None, None, 'controls,control')): False,
 ((None, None, None, 'kitchen'), (None, 'Kitchen knife', None, None)): False,
 ((None, None, None, 'banana slice'),
  (None, None, None, 'slice,slices')): False,
 ((None, 'Marine invertebrates', None, None),
  (None, 'Marine mammal', None, None)): False,
 ((None, 'Home appliance', None, None),
  (None, 'Kitchen appliance', None, None)): False,
 ((None, 'Door', None, 'door,doors'),
  (None, 'Door handle', None, 'door handle')): False,
 ((None, None, None, 'baseball'), (None, None, None, 'baseball cap')): False,
 (('traffic light', 'Traffic light', None, 'traffic light,traffic lights'),
  (None, None, None, 'traffic')): False,
 ((None, None, None, 'air'), (None, None, None, 'air conditioner')): False,
 ((None, 'Earrings', None, 'earrings,earring'),
  (None, None, None, 'bracelet,bracelets')): False,
 ((None, None, None, 'tennis court'), (None, None, None, 'tennis')): False,
 ((None, 'Belt', None, 'belt'), (None, 'Seat belt', None, None)): False,
 ((None, 'Sea turtle', None, None), (None, 'Sea lion', None, None)): False,
 (('train', 'Train', None, 'train,trains'),
  (None, None, None, 'train tracks,train track')): False,
 ((None, 'Tree', None, 'tree,trees'), (None, None, None, 'tree trunk')): False,
 ((None, None, None, 'water tank'), (None, None, None, 'gas tank')): False,
 ((None, None, None, 'stuffed animals,stuffed animal'),
  (None, 'Animal', None, 'animals,animal')): False,
 ((None, None, None, 'control panel'), (None, None, None, 'panel')): False,
 ((None, None, None, 'soccer ball'),
  (None, 'Ball', None, 'ball,balls')): False,
 ((None, None, None, 'burger'), (None, 'Hamburger', None, 'hamburger')): True,
 ((None, None, None, 'swimsuit'), (None, None, None, 'bikini')): False,
 ((None, 'Bat (Animal)', None, None),
  (None, 'Jaguar (Animal)', None, None)): False,
 ((None, 'Table', None, 'table,tables'),
  (None, None, None, 'end table')): False,
 (('skis', 'Ski', None, 'skis,ski'), (None, None, None, 'ski goggles')): False,
 ((None, 'Lamp', None, 'lamp,lamps'),
  (None, None, None, 'lampshade,lamp shade')): False,
 ((None, None, None, 'street'), (None, None, None, 'street sign')): False,
 ((None, None, None, 'bathroom'),
  (None, 'Bathroom accessory', None, None)): False,
 (('train', 'Train', None, 'train,trains'),
  (None, None, None, 'train car,train cars')): False,
 (('sandwich', 'Sandwich', None, 'sandwich,sandwhich'),
  (None, None, None, 'sandwiches')): True,
 ((None, None, None, 'sea'), (None, 'Sea lion', None, None)): False,
 ((None, 'Suit', None, 'suit'), (None, None, None, 'suit jacket')): False,
 ((None, None, None, 'stones,stone'), (None, None, None, 'stone wall')): False,
 ((None, None, None, 'tail fin'), (None, None, None, 'tail,tails')): False,
 (('sports ball', None, None, None), (None, 'Rugby ball', None, None)): False,
 ((None, None, None, 'pizza slice'),
  (None, None, None, 'slice,slices')): False,
 ((None, None, None, 'asparagus'),
  (None, 'Garden Asparagus', None, None)): True,
 ((None, None, 'ShowerDoor', 'shower door'),
  (None, 'Door', None, 'door,doors')): False,
 ((None, None, None, 'wires,wire'), (None, None, None, 'wire fence')): False,
 ((None, 'Piano', None, None), (None, 'Violin', None, None)): False,
 (('handbag', 'Handbag', None, 'handbag'), (None, None, None, 'purse')): True,
 ((None, None, 'Pot', 'pot,pots'), (None, None, None, 'coffee pot')): False,
 ((None, 'Ipod', None, 'ipod'), (None, None, None, 'iphone')): False,
 ((None, None, None, 'cabinet door'),
  (None, 'Bathroom cabinet', None, None)): False,
 ((None, 'Door', None, 'door,doors'), (None, None, None, 'door frame')): False,
 (('clock', 'Clock', None, 'clock,clocks'),
  (None, 'Wall clock', None, None)): False,
 (('dining table', 'Kitchen & dining room table', 'DiningTable', None),
  (None, None, None, 'picnic table')): False,
 ((None, None, None, 'light,lights'),
  (None, None, None, 'light fixture')): False,
 ((None, None, None, 'basket,baskets'),
  (None, 'Picnic basket', None, None)): False,
 ((None, 'Table', None, 'table,tables'),
  (None, None, None, 'picnic table')): False,
 (('cup', None, 'Cup', 'cup'), (None, 'Measuring cup', None, None)): False,
 ((None, 'Light switch', 'LightSwitch', 'light switch'),
  (None, None, None, 'switch')): False,
 (('bear', 'Bear', None, 'bear,bears'),
  (None, None, None, 'stuffed bear')): False,
 ((None, None, None, 'holder'), (None, None, None, 'candle holder')): False,
 ((None, None, None, 'shoe,shoes'),
  (None, None, None, 'tennis shoe,tennis shoes')): False,
 ((None, None, None, 'train car,train cars'),
  (None, None, None, 'passenger train')): False,
 ((None, None, None, 'toilet tank'), (None, 'Tank', None, 'tank')): False,
 ((None, None, None, 'spray'), (None, 'Hair spray', None, None)): False,
 ((None, None, None, 'soccer ball'), (None, 'Golf ball', None, None)): False,
 ((None, 'Tennis ball', None, 'tennis ball'),
  (None, 'Ball', None, 'ball,balls')): False,
 ((None, None, None, 'skateboard ramp'), (None, None, None, 'ramp')): False,
 ((None, None, None, 'meat'), (None, None, None, 'beef')): False,
 ((None, None, None, 'shoe,shoes'), (None, 'Footwear', None, None)): True,
 (('knife', 'Knife', 'Knife', 'knife,knives'),
  (None, 'Kitchen knife', None, None)): True,
 (('keyboard', 'Computer keyboard', None, 'keyboard,computer keyboard'),
  (None, 'Musical keyboard', None, None)): False,
 ((None, 'Burrito', None, None), (None, 'Taco', None, None)): False,
 ((None, 'Canoe', None, 'canoe'), (None, None, None, 'kayak')): False,
 ((None, 'Window', 'Window', 'window,windows'),
  (None, 'Window blind', None, None)): False,
 ((None, 'Towel', 'Towel', 'towel,towels'),
  (None, 'Paper towel', None, 'paper towels,paper towel')): False,
 ((None, None, None, 'area rug'), (None, None, None, 'rug')): True,
 ((None, None, None, 'bow'), (None, 'Bow and arrow', None, None)): False,
 ((None, 'Table', None, 'table,tables'),
  (None, 'Billiard table', None, None)): False,
 ((None, 'Ball', None, 'ball,balls'), (None, 'Golf ball', None, None)): False,
 ((None, None, None, 'child'), (None, None, None, 'children')): True,
 ((None, None, None, 'coin slot'), (None, None, None, 'slot')): False,
 ((None, None, 'ButterKnife', 'butter knife'),
  (None, None, None, 'butter')): False,
 ((None, None, None, 'trash'), (None, None, None, 'trash bag')): False,
 ((None, None, None, 'toilet tank'), (None, None, None, 'water tank')): False,
 (('motorcycle', 'Motorcycle', None, 'motorcycle,motorcycles'),
  (None, None, None, 'motorbike,motor bike')): True,
 ((None, None, None, 'rack'), (None, None, None, 'bike rack')): False,
 ((None, 'Tomato', 'Tomato', 'tomato,tomatoes'),
  (None, None, None, 'onions,onion')): False,
 (('broccoli', 'Broccoli', None, 'broccoli'),
  (None, 'Garden Asparagus', None, 'asparagus')): False,
 (('bicycle', 'Bicycle', None, 'bicycle,bicycles,bike,bikes'),
  (None, 'Bicycle helmet', None, None)): False,
 ((None, 'Light switch', 'LightSwitch', 'light switch'),
  (None, None, None, 'signal light')): False,
 ((None, None, None, 'clock tower'), (None, 'Wall clock', None, None)): False,
 ((None, None, None, 'swimsuit'), (None, 'Swimwear', None, None)): True,
 ((None, None, 'FloorLamp', 'floor lamp'),
  (None, None, None, 'table lamp')): False,
 ((None, None, 'TomatoSliced', 'tomato slice'),
  (None, None, None, 'pizza slice')): False,
 ((None, 'Ceiling fan', None, 'ceiling fan'),
  (None, None, None, 'ceiling')): False,
 ((None, None, None, 'cardboard box'), (None, None, None, 'boxes')): False,
 ((None, None, None, 'display case'), (None, None, None, 'display')): False,
 ((None, None, None, 'branches,tree branches,tree branch'),
  (None, None, None, 'branch')): True,
 ((None, 'Candle', 'Candle', 'candle,candles'),
  (None, None, None, 'candle holder')): False,
 ((None, 'Ball', None, 'ball,balls'), (None, None, None, 'ball cap')): False,
 (('baseball glove', 'Baseball glove', None, 'baseball glove'),
  (None, None, None, 'baseball mitt')): True,
 ((None, None, None, 'signal'), (None, None, None, 'traffic signal')): False,
 ((None, 'Vehicle', None, 'vehicle,vehicles'),
  (None, 'Land vehicle', None, None)): True,
 ((None, None, None, 'railroad tracks'),
  (None, None, None, 'railroad')): False,
 (('toilet', 'Toilet', 'Toilet', 'toilet'),
  (None, None, None, 'toilet bowl')): False,
 ((None, None, None, 'tennis player'),
  (None, None, None, 'baseball player,baseball players')): False,
 ((None, 'Garden Asparagus', None, 'asparagus'),
  (None, None, None, 'spinach')): False,
 (('teddy bear', 'Teddy bear', 'TeddyBear', 'teddy bear,teddy bears,teddy'),
  (None, None, None, 'stuffed bear')): True,
 ((None, 'Light switch', 'LightSwitch', 'light switch'),
  (None, None, None, 'light,lights')): False,
 ((None, None, None, 'coffee pot'),
  (None, 'Coffee cup', None, 'coffee cup')): False,
 ((None, None, 'FloorLamp', 'floor lamp'),
  (None, 'Lamp', None, 'lamp,lamps')): False,
 ((None, None, None, 'side mirror'), (None, None, None, 'side window')): False,
 ((None, None, None, 'tennis player'),
  (None, None, None, 'player,players')): False,
 (('car', 'Car', None, 'car,cars'), (None, None, None, 'police car')): False,
 ((None, None, None, 'toilet lid'), (None, None, None, 'lid')): False,
 ((None, 'Door handle', None, 'door handle'),
  (None, None, None, 'door frame')): False,
 (('toilet', 'Toilet', 'Toilet', 'toilet'),
  (None, None, None, 'toilet seat')): False,
 ((None, None, None, 'frame'), (None, None, None, 'door frame')): False,
 (('skateboard', 'Skateboard', None, 'skateboard,skate board,skateboards'),
  (None, None, None, 'skateboard ramp')): False,
 (('skis', 'Ski', None, 'skis,ski'), (None, None, None, 'ski lift')): False,
 (('carrot', 'Carrot', None, 'carrots,carrot'),
  (None, None, None, 'celery')): False,
 ((None, 'Skirt', None, 'skirt'), (None, None, None, 'blouse')): False,
 ((None, None, None, 'toilet brush'), (None, None, None, 'brush')): False,
 ((None, None, None, 'mountain,mountains'),
  (None, None, None, 'mountain top')): False,
 ((None, None, None, 'soccer ball'),
  (None, 'Cricket ball', None, None)): False,
 ((None, 'Trombone', None, None), (None, 'Trumpet', None, None)): False,
 ((None, 'Sofa bed', 'Sofa', 'sofa'), ('couch', 'Couch', None, 'couch')): True,
 ((None, None, None, 'rock wall'), (None, None, None, 'stone wall')): False,
 ((None, None, None, 'cap'), (None, None, None, 'ball cap')): False,
 ((None, 'Footwear', None, 'shoe,shoes'),
  (None, None, None, 'sneakers,sneaker')): False,
 ((None, None, None, 'rack'), (None, 'Spice rack', None, None)): False,
 (('tie', 'Tie', None, 'tie'), (None, None, None, 'necktie,neck tie')): True,
 (('bed', 'Bed', 'Bed', 'bed,beds'), (None, 'Infant bed', None, None)): False,
 ((None, None, None, 'men'), (None, None, None, 'women')): False,
 (('skis', 'Ski', None, 'skis,ski'),
  ('snowboard', 'Snowboard', None, 'snowboard,snow board')): False,
 ((None, None, None, 'rock wall'), (None, None, None, 'rocks,rock')): False,
 ((None, None, 'Pencil', 'pencil'), (None, 'Pencil case', None, None)): False,
 (('cat', 'Cat', None, 'cat,cats'), (None, None, None, 'kitten')): True,
 ((None, None, None, 'ski goggles'), (None, None, None, 'ski jacket')): False,
 ((None, None, None, 'coin slot'), (None, 'Coin', None, None)): False,
 ((None, 'Boot', 'Boots', 'boots,boot'),
  (None, None, None, 'ski boot,ski boots')): False,
 ((None, 'Cutting board', None, 'cutting board'),
  (None, None, None, 'board')): False,
 ((None, None, None, 'tennis player'),
  (None, 'Tennis ball', None, 'tennis ball')): False,
 (('toilet', 'Toilet', 'Toilet', 'toilet'),
  (None, None, None, 'toilet lid')): False,
                 }
def _maybe_merge(k1, k2):
    if (k1, k2) in manual_merges:
        return manual_merges[(k1, k2)]
    if (k2, k1) in manual_merges:
        return manual_merges[(k2, k1)]
    a = input()
    if a == '1':
        manual_merges[(k1, k2)] = True
    else:
        manual_merges[(k1, k2)] = False
    return manual_merges[(k1, k2)]


#####################################################################################
# Find the MOST SIMILAR THINGS round 2
for i in range(1000):
    mi = np.argmin(dist_mat)
    child = mi % dist_mat.shape[0]
    parent = mi // dist_mat.shape[0]

    if (category_to_name[parent] is None) or (category_to_name[child] is None):
        print("Uh oh recomputing")
        category_to_name = [x for x in category_to_name if x is not None]
        dist_mat = _compute_distance_matrix(category_to_name)

    if (category_to_name[child]['thor_name'] is not None) and (category_to_name[parent]['thor_name'] is not None):
        dist_mat[child, parent] = 9000.0
        dist_mat[parent, child] = 9000.0
        continue

    # CHECK FOR KEY PRESS

    print("Distance {:.3f}: Merging {} -> {}".format(dist_mat[parent, child],
                                                     category_to_name[child], category_to_name[parent]), flush=True)
    k_order = sorted(category_to_name[parent])
    k1 = tuple([category_to_name[parent][k] for k in k_order])
    k2 = tuple([category_to_name[child][k] for k in k_order])

    if _maybe_merge(k1, k2):
        _merge(parent, child)
    else:
        print("NOT MERGING", flush=True)
        dist_mat[child, parent] = 9000.0
        dist_mat[parent, child] = 9000.0

# Find nearby neighbors to all THOR objects
# for i, oi in enumerate(category_to_name[:126]):
#     nv = sum(int(oi[f'{k}_name'] is None) for k in ['coco', 'vg', 'oi'])
#     if nv < 2:
#         continue
#     most_sim = np.argsort(dist_mat[i])
#     for j in most_sim:
#         oj = category_to_name[j]
#         d = min(dist_mat[i, j], dist_mat[j, i])
#         if min(dist_mat[i, j], dist_mat[j, i]) < 0.3:
#             print("{}-{}. {:.3f} {}, {}".format(i, j, d, oi, category_to_name[j]), flush=True)
category_df = pd.DataFrame(category_to_name)
category_df.to_csv('all_categories.csv', index=False)

######################################

# Split the thor objects into training and test in a smart way.
thor_df = pd.read_csv('/home/rowan/code/ipk/data/object_counts.csv', skiprows=[0], index_col=0)
total_count = thor_df['TotalCount'].values
cols = [x for x in thor_df.columns if (thor_df[x].dtype == np.int64) and (not x.startswith(('TotalCount', 'salientMaterials')))]
thor_df_normed = thor_df[cols] / total_count[:, None]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=32, max_iter=300, verbose=10).fit(thor_df_normed.values)
thor_df['cluster_label'] = kmeans.labels_
for ind, df in thor_df.groupby('cluster_label'):
    print(df)

#
zero_shot = ['Blinds', 'Lettuce', 'LettuceSliced', 'Bread', 'BreadSliced', 'GarbageCan', 'Vase',
             'Mug', 'LightSwitch', 'SinkBasin', 'Kettle', 'Egg', 'EggCracked', 'Painting',
             'Pan', 'CellPhone', 'Mirror', 'Plate', 'Microwave', 'Window', 'Fridge', 'Sink', 'Drawer',
             'TeddyBear', 'Dumbbell', 'DeskLamp', 'CounterTop', 'WineBottle', 'Bed', 'Toaster', 'Potato', 'PotatoSliced']

category_df['is_zeroshot'] = category_df['thor_name'].apply(lambda x: x in zero_shot)


# Get synonyms
vico_words, vico_embs = get_vico(words_only=False)
vico_embs /= np.sqrt(np.square(vico_embs).sum(1) + 1e-8)[:, None]

banned_terms = []
w2i = {w:i for i, w in enumerate(vico_words)}
for _, row in category_df.iterrows():
    bt = []
    if row['is_zeroshot']:
        vs = [row[f'{k}_name'] for k in ['thor', 'coco', 'vg', 'oi']]
        vs = [x for x in vs if x is not None]
        vs = [y.lower() for x in vs for y in x.replace(' ', ',').split(',')]

        # Fix the THOR name
        extras = {
            'DeskLamp': ['lamp'],
            'SinkBasin': ['sink', 'basin'],
        }
        if row['thor_name'] in extras:
            vs.extend(extras[row['thor_name']])

        for w in vs:
            if w not in w2i:
                continue
            i = w2i[w]
            sims = vico_embs @ vico_embs[i]
            similar = np.where(sims >= 0.5)[0]
            for j in similar:
                bt.append(vico_words[j])
        bt = sorted(set(bt))
        banned_terms.append(','.join(bt))
    else:
        banned_terms.append(None)
category_df['preliminary_banned_terms'] = banned_terms

category_df.to_csv('../categories_and_banned_words.tsv', index=False, sep='\t')
