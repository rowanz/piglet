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


# Size delta: [z1, x1, width, height] -> new [z1, x1, w, h] correcting for the object being rotated to 0 degrees
object_name_to_size_delta = defaultdict(list)
object_type_to_size_delta = defaultdict(list)
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


def _get_zxwh_bb(obj):
    return np.array([obj['axisAlignedBoundingBox']['center']['z'],
                     obj['axisAlignedBoundingBox']['center']['x'],
                     obj['axisAlignedBoundingBox']['size']['z'],
                     obj['axisAlignedBoundingBox']['size']['x'],
                     ])


scene_names = sorted([x for x in env.controller.scenes_in_build if '_physics' in x])
for sn in tqdm(scene_names):
    env.reset(scene_name=sn)
    for i in range(20):
        env.randomize()
        env.randomize_agent_location()

        object_types = set([x['objectType'] for x in env.all_objects_with_properties({'openable': True})])
        for ot in object_types:
            env.controller.step(action='SetObjectStates', SetObjectStates={
                'objectType': ot,
                'stateChange': 'openable',
                'isOpen': False,
            })

        # Store state everywhere
        states0 = {x['objectId']: x for x in env.all_objects_with_properties({'isOpen': False, 'openable': True})}

        for ot in object_types:
            env.controller.step(action='SetObjectStates', SetObjectStates={
                'objectType': ot,
                'stateChange': 'openable',
                'isOpen': True,
            })

        states1 = {x['objectId']: x for x in env.all_objects_with_properties({'isOpen': True, 'openable': True})}

        # assert states0.keys() == states1.keys()

        for oid in states0:
            if oid not in states1:
                print("{} was in states0 but not in states1".format(oid), flush=True)
                continue

            name = states0[oid]['name']
            s0_box = _get_zxwh_bb(states0[oid])
            s1_box = _get_zxwh_bb(states1[oid])

            # get h/w offsets for s0 box such that both get spanned
            width_necessary = 2.0 * max(
                s1_box[0] - s0_box[0] + s1_box[2] / 2.0,
                s0_box[0] - s1_box[0] + s1_box[2] / 2.0,
                s0_box[2] / 2.0)
            height_necessary = 2.0 * max(
                s1_box[1] - s0_box[1] + s1_box[3] / 2.0,
                s0_box[1] - s1_box[1] + s1_box[3] / 2.0,
                s0_box[3] / 2.0)

            offsets = [width_necessary - s0_box[2], height_necessary - s0_box[3]]

            # Standardize offsets
            rotation = round_to_factor(states0[oid]['rotation']['y'], 15) % 360
            if rotation not in (0, 90, 180, 270):
                continue
            # assert rotation == (round_to_factor(states1[oid]['rotation']['y'], 90) % 360)
            if rotation in (90, 270):
                offsets = [offsets[1], offsets[0]]

            if name in object_name_to_size_delta:
                print("{}: ({:.3f},{:.3f}) vs {:.3f},{:.3f})".format(name, *object_name_to_size_delta[name][0],
                                                                     *offsets,
                                                                     ), flush=True)
            object_name_to_size_delta[name].append(offsets)
            object_type_to_size_delta[states0[oid]['objectType']].append(offsets)

            # delta = s1_box - s0_box

object_name_to_size_delta_dict = {name: (np.percentile(np.array(sizes)[:,0], 99),
                                         np.percentile(np.array(sizes)[:,1], 99)) for name, sizes in object_name_to_size_delta.items()}

object_type_to_size_delta_dict = {name: (np.percentile(np.array(sizes)[:,0], 80),
                                         np.percentile(np.array(sizes)[:,1], 80)) for name, sizes in object_type_to_size_delta.items()}


with open('../data/size_deltas.json', 'w') as f:
    json.dump({'object_name_to_size_delta': object_name_to_size_delta_dict, 'object_type_to_size_delta':
               object_type_to_size_delta_dict}, f)
