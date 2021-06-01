"""
Get size mulitpliers when objects open
"""
import sys

sys.path.append('../')
from sampler.ai2thor_env import AI2ThorEnvironment, round_to_factor
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
import random
import os

from datetime import datetime


def rotation_angle_to_object(object_info, point):
    """
    Given a point, compute the best rotation angle to object
      the coordinate system is a bit weird

                  90 degrees
                       ^ +x

      180 degrees             ------> +z     0 degrees
                       |
                       |
                       V
                   270 degrees


    :param object_info:
    :param point: [x,y,z]
    :return:
    """
    object_coords = object_info['axisAlignedBoundingBox']['center'] if 'axisAlignedBoundingBox' in object_info else \
        object_info['position']

    x_delta = object_coords['x'] - point['x']
    z_delta = object_coords['z'] - point['z']
    r = np.sqrt(np.square(x_delta) + np.square(z_delta))

    angle = np.arctan2(x_delta / r, z_delta / r) * 180 / np.pi

    if angle < 0:
        angle += 360
    if angle > 360.0:
        angle -= 360.0
    return angle


def horizon_angle_to_object(object_info, point):
    """
           ^     - angle
          /
        me                       reference pt
          \
           V     + angle

            we're facing the object already

    :param object_info:
    :param point: [x,y,z]
    :return:
    """
    object_coords = object_info['axisAlignedBoundingBox']['center'] if 'axisAlignedBoundingBox' in object_info else \
        object_info['position']

    my_height = 1.575

    y_delta = object_coords['y'] - my_height
    xz_dist = np.sqrt(np.square(object_coords['x'] - point['x']) + np.square(object_coords['z'] - point['z']))
    r = np.sqrt(np.square(xz_dist) + np.square(y_delta))

    angle = np.arctan2(-y_delta / r, xz_dist / r) * 180 / np.pi

    if angle < 0:
        angle += 360
    if angle > 360.0:
        angle -= 360.0
    return angle


knob_to_burner = defaultdict(lambda: defaultdict(int))
env_args = {
    "player_screen_height": 384,
    "player_screen_width": 640,
    "quality": "Very Low",
    'x_display': '0.0',
}
env = AI2ThorEnvironment(
    make_agents_visible=True,
    object_open_speed=0.05,
    restrict_to_initially_reachable_points=True,
    render_class_image=False,
    render_object_image=False,
    **env_args)


def _all_off():
    object_types = set(
        [x['objectType'] for x in env.all_objects_with_properties({'toggleable': True, 'isToggled': True})])
    for ot in object_types:
        env.controller.step(action='SetObjectStates', SetObjectStates={
            'objectType': ot,
            'stateChange': 'toggleable',
            'isToggled': False,
        })


def _delete_objects():
    for x in env.all_objects_with_properties({'pickupable': True}):
        if x['objectType'] not in ('StoveKnob', 'StoveBurner'):
            env.controller.step(action='RemoveFromScene', objectId=x['objectId'])

def _dump(knob_to_burner):
    knob_to_burner = {k: sorted(v.keys(), key=lambda k: -v[k])[0] for k, v in knob_to_burner.items()}
    with open("../data/knob_to_burner.json", 'w') as f:
        json.dump(knob_to_burner, f)

if os.path.exists("../data/knob_to_burner.json"):
    with open("../data/knob_to_burner.json", 'r') as f:
        for k, b in json.load(f).items():
            knob_to_burner[k][b] += 100


scene_names = sorted([x for x in env.controller.scenes_in_build if '_physics' in x])
for sn in tqdm(scene_names):

    env.reset(scene_name=sn)
    knobs = [x for x in env.all_objects_with_properties({'objectType': 'StoveKnob'}) if x['name'] not in knob_to_burner]
    if len(knobs) == 0:
        continue
    for j in range(100):
        env.reset(scene_name=sn)
        env.randomize()
        _delete_objects()
        _all_off()
        env.randomize_agent_location()
        knobs = [x for x in env.all_objects_with_properties({'objectType': 'StoveKnob'}) if x['name'] not in knob_to_burner]
        if len(knobs) == 0:
            break
        knob = random.choice(knobs)

        # Turn it on
        # FORCEVISIBLE DOESNT WORK SO WE NEED TO TELEPORT SOMEWHERE NEARBY

        new_pos = sorted(env.initially_reachable_points, key=lambda x: env.position_dist(x, knob['position']))[0]
        new_pos = {k: new_pos[k] for k in 'xyz'}
        new_pos['rotation'] = rotation_angle_to_object(knob, new_pos)
        new_pos['horizon'] = horizon_angle_to_object(knob, new_pos)

        env.teleport_agent_to(**new_pos, ignore_y_diffs=True, only_initially_reachable=False)

        event = env.controller.step(action='ToggleObjectOn', objectId=knob['objectId'])
        env.step({'action': 'Pass'})
        env.step({'action': 'Pass'})
        env.step({'action': 'Pass'})
        env.step({'action': 'Pass'})
        env.step({'action': 'Pass'})

        if event.metadata['lastActionSuccess']:
            on_burner = env.all_objects_with_properties({'objectType': 'StoveBurner', 'ObjectTemperature': 'Hot'})
            event = env.controller.step(action='ToggleObjectOff', objectId=knob['objectId'])

            to_dump = len(knob_to_burner[knob['name']]) == 0
            for b in on_burner:
                knob_to_burner[knob['name']][b['name']] += 1

            if to_dump:
                _dump(knob_to_burner)
                print("NEW KNOB!! {}".format(datetime.now().strftime("%H:%M:%S")), flush=True)
            elif j == 0:
                _dump(knob_to_burner)