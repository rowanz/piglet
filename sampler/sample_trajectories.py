"""

Sample trajectories
"""
import sys

sys.path.append('../')
from sampler.ai2thor_env import AI2ThorEnvironment, VISIBILITY_DISTANCE, round_to_factor
import random
import numpy as np
from sampler.data_utils import GCSH5Writer
import json
from tqdm import tqdm
import os
import pandas as pd
import argparse
import glob

parser = argparse.ArgumentParser(description='Sample Trajectories')
parser.add_argument(
    '-display',
    dest='display',
    default=0,
    type=int,
    help='which display we are on'
)
parser.add_argument(
    '-seed',
    dest='seed',
    default=123456,
    type=int,
    help='seed to use'
)
parser.add_argument(
    '-out_path',
    dest='out_path',
    type=str,
    help='Base path to use.'
)
parser.add_argument(
    '-nex',
    dest='nex',
    default=10000,
    type=int,
    help='Num examples to generate'
)
args = parser.parse_args()


with open(os.path.join(os.path.dirname(os.path.join(__file__)), '..', 'data', 'size_deltas.json'), 'r') as f:
    SIZE_DELTAS = json.load(f)
with open(os.path.join(os.path.dirname(os.path.join(__file__)), '..', 'data', 'knob_to_burner.json'), 'r') as f:
    KNOB_TO_BURNER = json.load(f)

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


def clip_angle_to_nearest(true_angle, possible_angles=(0, 90, 180, 270)):
    """
    Clips angle to the nearest one
    :param true_angle: float
    :param possible_angles: other angles
    :return:
    """
    pa = np.array(possible_angles)
    dist = np.abs(pa - true_angle)
    dist = np.minimum(dist, 360.0 - dist)
    return int(pa[np.argmin(dist)])


def get_object_zxwh_and_expansion(target_object, openable_object_pad_distance=0.0, object_pad_distance=0.0):
    """
    Compute object bounding box coordinates, with possible expansion

    :param target_object:
    :param openable_object_pad_distance: pad openable objects that are currently closed this much EXTRA
    :param object_pad_distance: pad all objects this much
    :return:
    """
    object_zxwh = np.array([target_object['axisAlignedBoundingBox']['center']['z'],
                            target_object['axisAlignedBoundingBox']['center']['x'],
                            target_object['axisAlignedBoundingBox']['size']['z'],
                            target_object['axisAlignedBoundingBox']['size']['x'],
                            ])
    object_zxwh2 = np.copy(object_zxwh)

    # Increase size a bit accordingly
    if target_object['openable'] and not target_object['isOpen']:
        sd = SIZE_DELTAS['object_name_to_size_delta'].get(target_object['name'],
                                                          SIZE_DELTAS['object_type_to_size_delta'].get(
                                                              target_object['objectType'], [0.0, 0.0]))

        # Rotation matrix
        theta = target_object['rotation']['y'] * np.pi / 180
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        sd = np.abs(R @ np.array(sd)) + openable_object_pad_distance # Agent radius is tehcnically 0.2 (I dealt with that elsewhere) but 0.4 seems
                                            # needed for some annoying objects
        object_zxwh2[2:] += sd

    object_zxwh2[2:] += object_pad_distance # agent radius is 0.2
    return object_zxwh, object_zxwh2


def zxwh_to_zzxx(bbox_xywh):
    sizes = bbox_xywh[..., 2:] / 2.0
    return np.concatenate((bbox_xywh[..., :2] - sizes, bbox_xywh[..., :2] + sizes), -1)


def interval_pos(test_x, x0, x1):
    """
    In 1D space:

                  x0                  x1
    ##############  ##################  ##############
    # ---- position            0 position          + position

    :param test_x:
    :param x0:
    :param x1:
    :return:
    """
    assert x1 >= x0

    if test_x < x0:
        return test_x - x0
    if test_x > x1:
        return test_x - x1
    return 0.0


def distance_to_object_manifold(agent_pos, target_object, openable_object_pad_distance=0.0, object_pad_distance=0.0):
    """
    Distance from the agent to the object manifold. If the object is openable we'll pretend it's open and hallucinate accordingly

    For simplicitly we'll just use 2d distances

    The agent is a circle around its position with radius 0.2

    :param agent_pos:
    :param object:
    :return:
    """

    if target_object['objectType'] == 'Floor':
        return 0.0

    # Just return the expanded box.
    _, object_zxwh = get_object_zxwh_and_expansion(target_object,
                                                   openable_object_pad_distance=openable_object_pad_distance,
                                                   object_pad_distance=object_pad_distance)

    # Compute distance to the manifold. negative distance if INSIDE.
    # Check if we're inside
    object_zzxx = zxwh_to_zzxx(object_zxwh)

    z_dist = interval_pos(agent_pos['z'], object_zzxx[0], object_zzxx[2])
    x_dist = interval_pos(agent_pos['x'], object_zzxx[1], object_zzxx[3])

    if z_dist == 0.0 and x_dist == 0.0:
        # Inside the boundary?
        return -0.01

    r = np.sqrt(np.square(z_dist) + np.square(x_dist))
    return r


# Subactions. they aren't DOING anything yet just planning ======================================
def path_to_object(env: AI2ThorEnvironment, target_object, angle_noise=10, dist_noise=0.1, dist_to_obj_penalty=2.0,
                   faces_good_side_penalty=10.0):
    """
    Go to the object (starting from the Agent's location)
    :param env:
    :param target_object:
    :param angle_noise:
    :param dist_noise: standard deviation for noise we'll add to shortest pathfinder
    :param dist_to_obj_penalty:
    :return:
    """
    reachable_pts = env.currently_reachable_points

    start_location = env.get_agent_location()

    receptacle = None
    if target_object['openable'] and not target_object['isOpen']:
        receptacle = target_object
    for pr in target_object['parentReceptacles']:
        mr = env.get_object_by_id(pr)
        if mr['openable'] and not mr['isOpen']:
            receptacle = mr

    # Find all pts
    reachable_points_touching_object = []
    for pt in reachable_pts:

        ctr_dist = env.position_dist(pt, target_object['position'])
        pt_dist = distance_to_object_manifold(pt, target_object, object_pad_distance=0.0)

        if receptacle is not None:
            # I really don't know what to set this to safely
            oopd = random.random()
            pt_dist_to_receptacle = distance_to_object_manifold(pt, receptacle,
                                                                openable_object_pad_distance=oopd,
                                                                object_pad_distance=0.2)
            if pt_dist_to_receptacle < 0.0:
                continue

        vd = 1.49
        if target_object['objectType'] in ('Fridge',):
            vd += 0.5

        if (ctr_dist > vd) and (pt_dist > vd):
            continue

        # Might need a minimum distance away from the object

        # Get angle
        ra = rotation_angle_to_object(target_object, pt) + float(np.random.uniform(-angle_noise, angle_noise))
        ra = clip_angle_to_nearest(ra)

        ha = horizon_angle_to_object(target_object, pt) + float(np.random.uniform(-angle_noise, angle_noise))
        ha = clip_angle_to_nearest(ha, [-30, 0, 30, 60])

        pt2 = {k: v for k, v in pt.items()}
        pt2['horizon'] = ha
        pt2['rotation'] = ra
        pt2['dist'] = pt_dist
        pt2['dist_to_me'] = env.position_dist(start_location, pt)

        # Check if we're facing the good side of the object for an object like fridge
        pt2['faces_good_side'] = True
        if receptacle is not None:
            obj, obj_expanded = get_object_zxwh_and_expansion(receptacle,
                                                              openable_object_pad_distance=0.2,
                                                              object_pad_distance=0.2)

            object_zzxx = zxwh_to_zzxx(obj_expanded)
            # Figure out what quadrant we're in
            z_dist = interval_pos(pt2['z'], object_zzxx[0], object_zzxx[2])
            x_dist = interval_pos(pt2['x'], object_zzxx[1], object_zzxx[3])

            # Z expansion, X expansion.
            size_delta_z, size_delta_x = (obj_expanded - obj)[2:]

            if (abs(z_dist) > 0) and (abs(x_dist) > 0):
                pt2['faces_good_side'] = False
            else:
                # If X expansion is a lot longer, we want Z cordinates to be 0 dist and X coordinates to be + dist
                if size_delta_x > (size_delta_z + 0.25):
                    pt2['faces_good_side'] = (z_dist == 0.0)
                if size_delta_z > (size_delta_x + 0.25):
                    pt2['faces_good_side'] = (x_dist == 0.0)

        reachable_points_touching_object.append(pt2)

    if len(reachable_points_touching_object) == 0:
        raise ValueError("No points touching object {}".format(target_object['objectId']))

    dists = np.array(
        [(pt2['dist'], pt2['dist_to_me'], float(pt2['faces_good_side'])) for pt2 in reachable_points_touching_object])

    # 1 standard dist -> 0.25
    joint_dist = dist_to_obj_penalty * dists[:, 0] + dists[:, 1] + np.random.randn(
        dists.shape[0]) * dist_noise + faces_good_side_penalty * (1.0 - dists[:, 2])

    pt = reachable_points_touching_object[int(np.argmin(joint_dist))]

    res = env.get_fancy_shortest_path(start_location, pt)
    if res is None:
        raise ValueError("Shortest path failed")
    return res


def objects_equal(obj1, obj2):
    def _bbox_equal(b1, b2, tolerance=4e-2):
        b1_np = np.array(b1['cornerPoints'])
        b2_np = np.array(b2['cornerPoints'])
        return np.abs(b1_np - b2_np).max() < tolerance

    def _xyz_equal(b1, b2, tolerance=4e-2, mod360=False):
        p1_np = np.array([b1[k2] for k2 in 'xyz'])
        p2_np = np.array([b2[k2] for k2 in 'xyz'])
        dist = np.abs(p1_np - p2_np)
        if mod360:
            dist = np.minimum(dist, 360.0 - dist)

        return dist.max() < tolerance

    if obj1 == obj2:
        return True

    if obj1.keys() != obj2.keys():
        print("Keys differ", flush=True)
        return False

    keys = sorted(obj1.keys())
    for k in keys:
        if obj1[k] == obj2[k]:
            continue

        # Float distance
        if k == 'position':
            if not _xyz_equal(obj1[k], obj2[k]):
                # print(f"{k} - {obj1[k]} {obj2[k]}")
                return False
        elif k == 'rotation':
            if not _xyz_equal(obj1[k], obj2[k], mod360=True, tolerance=1.0):
                # print(f"{k} - {obj1[k]} {obj2[k]}")
                return False
        elif k == 'axisAlignedBoundingBox':
            for k3 in ['center', 'size']:
                if not _xyz_equal(obj1[k][k3], obj2[k][k3]):
                    # print(f"AxisAlignedBoundingBox -> {k3} ({obj1['objectId']})")
                    return False
        elif k in ('isMoving', 'distance', 'visible'):
            continue
        else:
            # print(k, flush=True)
            return False
    return True


def randomize_toggle_states(env: AI2ThorEnvironment, random_prob=0.9):
    """
    Randomize toggle states. unfortunately this does this uniformly for the entire scene
    :param env:
    :param random_prob:
    :return:
    """
    possible_object_attributes = {
        # 'openable': ('isOpen', (True, False)),
        'toggleable': ('isToggled', (True, False)),
        'breakable': ('isBroken', (True, False)),
        'canFillWithLiquid': ('isFilledWithLiquid', (True, False)),
        'dirtyable': ('isDirty', (True, False)),
        'cookable': ('isCooked', (True, False)),
        'sliceable': ('isSliced', (True, False)),
        'canBeUsedUp': ('isUsedUp', (True, False)),
    }

    possible_actions_to_vals = {}
    for obj in env.all_objects():
        overlapping_properties = [(k, v[0], v[1]) for k, v in possible_object_attributes.items() if obj.get(k, False)]

        for k, v0, v1 in overlapping_properties:
            possible_actions_to_vals[(obj['objectType'], k, v0)] = v1

    for k, v in possible_actions_to_vals.items():
        if random.random() > random_prob:
            new_state = random.choice(v)
            env.controller.step(action='SetObjectStates', SetObjectStates={
                'objectType': k[0],
                'stateChange': k[1],
                k[2]: new_state,
            })

def break_all(env: AI2ThorEnvironment, object_type):
    env.controller.step(action='SetObjectStates', SetObjectStates={
        'objectType': object_type,
        'stateChange': 'breakable',
        'isBroken': True,
    })

def fill_all_objects_with_liquid(env: AI2ThorEnvironment):
    empty_objects = set()
    empty_object_types = set()
    for obj in env.all_objects():
        if obj['canFillWithLiquid'] and not obj['isFilledWithLiquid']:
            empty_objects.add(obj['objectId'])
            empty_object_types.add(obj['objectType'])
    #
    # for ot in empty_object_types:
    #     env.controller.step(action='SetObjectStates', SetObjectStates={
    #         'objectType': ot,
    #         'stateChange': 'canFillWithLiquid',
    #         'isFilledWithLiquid': True,
    #     })

    for obj in empty_objects:
        liquid = random.choice(['coffee', 'wine', 'water'])
        env.controller.step({'action': 'FillObjectWithLiquid', 'objectId': obj, 'fillLiquid': liquid,
                             'forceVisible': True})


def close_everything(env: AI2ThorEnvironment):
    object_types_open = set()
    for obj in env.all_objects():
        if obj['openable'] and obj['isOpen']:
            object_types_open.add(obj['objectType'])

    for x in object_types_open:
        env.controller.step(action='SetObjectStates', SetObjectStates={
            'objectType': x,
            'stateChange': 'openable',
            'isOpen': False,
        })

def retract_hand(env: AI2ThorEnvironment):
    """
    Move the hand back so it's not super visible. this will be reset.
    :param env:
    :return:
    """
    oih = env.object_in_hand()
    if oih is None:
        return
    # current_rotation = clip_angle_to_nearest(env.get_agent_location()['rotation'])
    env.controller.step(action='MoveHandDown', moveMagnitude=0.2)
    env.controller.step(action='MoveHandBack', moveMagnitude=0.1)

def group_knobs_and_burners(env: AI2ThorEnvironment):
    """
    We'll group together stoves and knobs that go together
    :param env:
    :return:
    """
    stoveburners = {x['name']: x for x in env.all_objects_with_properties({'objectType': 'StoveBurner'})}
    knobs = {x['name']: x for x in env.all_objects_with_properties({'objectType': 'StoveKnob'})}

    knobs_and_burners = []
    for k, v in knobs.items():
        if k not in KNOB_TO_BURNER:
            print(f"{k} not found???")
            continue
        bn = KNOB_TO_BURNER[k]
        if bn in stoveburners:
            knobs_and_burners.append((v, stoveburners[bn]))
        else:
            print("Oh no burner {} not found?".format(bn), flush=True)
    return knobs_and_burners


class RecordingEnv(object):
    """
    Record
    # this class needs to be able to record the path:
    # 1. actions
    # 2. images
    # 3. bounding boxes, including object attrs
    """

    def __init__(self, env: AI2ThorEnvironment, text='', main_object_ids=(), ):

        self.env = env

        # Things we're recording
        # "inputs"
        self.meta_info = None

        self.frames = []
        self.object_id_to_states = {}  # objid -> {t} -> obj. Default prior is that states don't change
        self.bboxes = []
        self.agent_states = []

        # "outputs"
        self.output_actions = []
        self.output_action_results = []

        self.alias_object_id_to_old_object_id = {}

        self.meta_info = {
            'scene_name': self.env.scene_name,
            'text': text,
            'main_object_ids': main_object_ids,
            'width': env._start_player_screen_width,
            'height': env._start_player_screen_height,
        }
        self.log_observations()


    def __len__(self):
        return len(self.frames)

    @property
    def new_items(self):
        return {k: v[len(self) - 1] for k, v in self.object_id_to_states.items() if (len(self) - 1) in v}

    def step(self, action_dict, action_can_fail=False, record_failure=True):
        """
        :param action_dict:
        :param action_can_fail: Whether we're OK with the action failing
        :return:
        """

        res = self.env.step(action_dict)

        if (not self.env.last_action_success) and (not action_can_fail):
            raise ValueError("Action {} failed, {}".format(action_dict, self.env.last_event.metadata['errorMessage']))

        if (not record_failure) and (not self.env.last_action_success):
            return False

        self.output_actions.append(action_dict)
        self.output_action_results.append({
            'action_success': self.env.last_action_success,
            'action_err_msg': self.env.last_event.metadata['errorMessage'],
        })
        self.log_observations()
        return self.env.last_action_success

    def log_observations(self):
        t = len(self.frames)
        # frames
        self.frames.append(self.env.last_event.frame)

        # Update object states
        for obj in self.env.all_objects():
            # We must have added a new object
            if obj['objectId'] not in self.object_id_to_states:
                # if (t > 0) and (not 'Slice' in obj['name']):
                #     import ipdb
                #     ipdb.set_trace()
                self.object_id_to_states[obj['objectId']] = {t: obj}
            else:
                last_t = sorted(self.object_id_to_states[obj['objectId']].keys())[-1]

                # Object changed
                if not objects_equal(self.object_id_to_states[obj['objectId']][last_t], obj):
                    self.object_id_to_states[obj['objectId']][t] = obj

        # Bounding boxes
        self.bboxes.append({k: v.tolist() for k, v in self.env.last_event.instance_detections2D.items()})
        self.agent_states.append(self.env.get_agent_location())

    def save(self, fn='temp.h5'):
        writer = GCSH5Writer(fn)

        # Get object
        object_ids = sorted(self.object_id_to_states)
        object_id_to_ind = {id: i for i, id in enumerate(object_ids)}

        # No need to store identities
        for k, v in sorted(self.alias_object_id_to_old_object_id.items()):
            if k == v:
                self.alias_object_id_to_old_object_id.pop(k)

        writer.create_dataset('alias_object_id_to_old_object_id',
                              data=np.string_(json.dumps(self.alias_object_id_to_old_object_id).encode('utf-8')))

        writer.create_dataset('object_ids', np.string_(object_ids))
        writer.create_dataset('frames', data=np.stack(self.frames), compression='gzip', compression_opts=9)

        # Get bounding box -> ind
        # [ind, box_x1, box_y1, box_x2, box_y2]
        bbox_group = writer.create_group('bboxes')
        for t, bbox_dict_t in enumerate(self.bboxes):
            # NOTE: This filters out some objects.
            bbox_list = [[object_id_to_ind[k]] + v for k, v in bbox_dict_t.items() if k in object_id_to_ind]
            bbox_array = np.array(sorted(bbox_list, key=lambda x: x[0]), dtype=np.uint16) ###### oh no I was saving this as uint8 earlier and things were clipping. UGH
            bbox_group.create_dataset(f'{t}', data=bbox_array)

        # Position / Rotation / center / size
        xyz_coords = writer.create_group('pos3d')
        for k in sorted(self.object_id_to_states):
            xyz_coords_k = xyz_coords.create_group(k)
            for t in sorted(self.object_id_to_states[k]):
                coords_to_use = [self.object_id_to_states[k][t].pop('position'),
                                 self.object_id_to_states[k][t].pop('rotation'),
                                 self.object_id_to_states[k][t]['axisAlignedBoundingBox'].pop('center'),
                                 self.object_id_to_states[k][t]['axisAlignedBoundingBox'].pop('size')]

                xyz_coords_kt_np = np.array([[p[k2] for k2 in 'xyz'] for p in coords_to_use], dtype=np.float32)
                xyz_coords_k.create_dataset(f'{t}', data=xyz_coords_kt_np)
                self.object_id_to_states[k][t].pop('axisAlignedBoundingBox')
                self.object_id_to_states[k][t].pop('objectOrientedBoundingBox')

        # print("{} total object things".format(sum([len(v) for v in self.object_id_to_states.values()])))
        writer.create_dataset('object_id_to_states', data=np.string_(
            json.dumps(self.object_id_to_states).encode('utf-8'),
        ))

        writer.create_dataset('agent_states', data=np.array([[s[k] for k in ['x', 'y', 'z', 'rotation', 'horizon']]
                                                             for s in self.agent_states], dtype=np.float32))

        writer.create_dataset('output_actions', data=np.string_(json.dumps(self.output_actions).encode('utf-8')))
        writer.create_dataset('output_action_results',
                              data=np.string_(json.dumps(self.output_action_results).encode('utf-8')))
        writer.create_dataset('meta_info', data=np.string_(json.dumps(self.meta_info).encode('utf-8')))
        writer.close()

#############################################################################


def inch_closer_path(renv: RecordingEnv, target_receptacle):
    """
    Assuming the receptacle is open, scoot a bit closer. Keep existing horizon and rotation.
    :param env:
    :param target_receptacle:
    :return:
    """
    env.recompute_reachable()
    reachable_pts = renv.env.currently_reachable_points

    start_location = renv.env.get_agent_location()
    start_location['horizon'] = clip_angle_to_nearest(start_location['horizon'], [-30, 0, 30, 60])
    start_location['rotation'] = clip_angle_to_nearest(start_location['rotation'])
    start_dist = renv.env.position_dist(start_location, target_receptacle['position'])

    closer_pts = []
    for pt in reachable_pts:
        pt_dist = renv.env.position_dist(pt, target_receptacle['position'])
        if pt_dist >= (start_dist - 0.05):
            continue
        pt2 = {k: v for k, v in pt.items()}
        pt2['dist'] = pt_dist
        pt2['dist_to_me'] = renv.env.position_dist(start_location, pt)

        pt2['horizon'] = start_location['horizon']
        if pt2['horizon'] != clip_angle_to_nearest(horizon_angle_to_object(target_receptacle, pt), [-30, 0, 30, 60]):
            continue

        pt2['rotation'] = start_location['rotation']
        if pt2['rotation'] != clip_angle_to_nearest(rotation_angle_to_object(target_receptacle, pt)):
            continue

        closer_pts.append(pt2)
    if len(closer_pts) == 0:
        return []

    dists = np.array([(pt2['dist'], pt2['dist_to_me']) for pt2 in closer_pts])
    score = 20.0 * dists[:, 0] + dists[:, 1] + 0.5 * np.random.randn(dists.shape[0])
    pt = closer_pts[int(np.argmin(score))]

    forward_path = env.get_fancy_shortest_path(start_location, pt, fix_multi_moves=False, num_inner_max=4)
    if forward_path is None:
        return []
    forward_path = [x for x in forward_path if not x['action'].startswith('Look')]

    if any([x['action'] in ('JumpAhead', 'RotateLeft', 'RotateRight') for x in forward_path]):
        forward_path = []

    # Now we can have a reverse path too
    antonyms = {'MoveAhead': 'MoveBack', 'MoveLeft': 'MoveRight', 'MoveRight': 'MoveLeft'}
    reverse_path = []
    for p in forward_path[::-1]:
        reverse_path.append({
            'action': antonyms[p['action']],
            '_start_state_key': p['_end_state_key'],
            '_end_state_key': p['_start_state_key'],
        })

    for i, p in enumerate(forward_path):
        good = renv.step(p, action_can_fail=True, record_failure=False)
        if not good:
            my_key = env.get_key(env.get_agent_location())
            key_matches = np.array([rp['_start_state_key'] == my_key for rp in reverse_path])
            if not np.any(key_matches):
                reverse_path = []
            else:
                ki = int(np.where(key_matches)[0][0])
                reverse_path = reverse_path[ki:]
            break
    return reverse_path


def pickup_object(renv: RecordingEnv, target_object_id, navigate=True,
                  action_can_fail=False, force_visible=False):
    """
    :param renv:
    :param target_object_id:
    :param navigate: Whether to navigate there
    :param action_can_fail: Whether the final pickup action can fail
    :return:
    """

    target_object = renv.env.get_object_by_id(target_object_id)

    if target_object is None:
        raise ValueError("No target object")

    # Check if any objects are held, drop if so
    held_object = renv.env.object_in_hand()
    if held_object is not None:
        raise ValueError("held object??")

    must_open = False
    if target_object['parentReceptacles'] is not None and len(target_object['parentReceptacles']) > 0:
        pr = env.get_object_by_id(target_object['parentReceptacles'][0])
        if pr['openable'] and not pr['isOpen']:
            must_open = True
    else:
        pr = None

    if navigate:
        d2op = (random.random()*0.5 - 0.25) if must_open else random.random()
        fgsp = 10.0 if random.random() > 0.5 else 0.0
        for p in path_to_object(renv.env, target_object=target_object, dist_to_obj_penalty=d2op, faces_good_side_penalty=fgsp):
            renv.step(p, action_can_fail=False)

    # Inch closer
    if must_open and (pr is not None):
        renv.step({'action': 'OpenObject', 'objectId': pr['objectId']}, action_can_fail=False)

        reverse_path = inch_closer_path(renv, pr)
    else:
        pr = None
        reverse_path = []

    renv.step({'action': 'PickupObject', 'objectId': target_object_id, 'forceVisible': force_visible}, action_can_fail=action_can_fail)

    for p in reverse_path:
        good = renv.step(p, action_can_fail=True, record_failure=False)
        if not good:
            break

    if (pr is not None) and (pr['openable'] and not pr['isOpen']):
        renv.step({'action': 'CloseObject', 'objectId': pr['objectId']}, action_can_fail=False)
    return True


def put_object_in_receptacle(renv: RecordingEnv, target_receptacle_id, navigate=True, close_receptacle=True,
                             action_can_fail=False):
    """
    :param renv:
    :param target_receptacle:
    :return:
    """
    target_receptacle = renv.env.get_object_by_id(target_receptacle_id)

    # Sanity check we're holding an object
    held_object = renv.env.object_in_hand()
    if held_object is None:
        if action_can_fail:
            return False
        raise ValueError("No held object")

    if target_receptacle is None:
        raise ValueError("No receptacle?")

    # Don't go too close to the object if it's open
    must_open = target_receptacle['openable'] and not target_receptacle['isOpen']
    d2op = (random.random() * 0.5 - 0.25) if must_open else random.random()
    if navigate:
        fgsp = 10.0 if random.random() > 0.5 else 0.0
        path = path_to_object(renv.env, target_object=target_receptacle, dist_to_obj_penalty=d2op, faces_good_side_penalty=fgsp)
        for p in path:
            renv.step(p, action_can_fail=False)

    # Open if needed
    if must_open:
        renv.step({'action': 'OpenObject', 'objectId': target_receptacle_id}, action_can_fail=False)
        reverse_path = inch_closer_path(renv, renv.env.get_object_by_id(target_receptacle_id))
    else:
        reverse_path = []

    # Move hand back
    if not env.get_object_by_id(held_object['objectId'])['visible']:
        retract_hand(renv.env)

    succeeds = renv.step(
        {'action': 'PutObject', 'objectId': held_object['objectId'], 'receptacleObjectId': target_receptacle_id},
        action_can_fail=action_can_fail)

    for p in reverse_path:
        good = renv.step(p, action_can_fail=True, record_failure=False)
        if not good:
            break

    if succeeds and close_receptacle and target_receptacle['openable']:
        succeeds = renv.step({'action': 'CloseObject', 'objectId': target_receptacle_id}, action_can_fail=False)
    return succeeds


def pour_out_liquid(renv: RecordingEnv, action_can_fail=True):
    """
    Assuming we're holding stuff -- pour it out
    :param renv:
    :param action_can_fail:
    :return:
    """
    held_obj = renv.env.object_in_hand()
    if held_obj is None:
        if not action_can_fail:
            raise ValueError("NO held obj")
        return False

    # Option 1: Rotate 180 degrees upside down
    if random.random() < 0.5:
        renv.step({'action': 'RotateHand', 'x': 180}, action_can_fail=action_can_fail)
        renv.step({'action': 'RotateHand', 'x': 0}, action_can_fail=action_can_fail)
    else:
        if held_obj['isFilledWithLiquid']:
            renv.step({'action': 'EmptyLiquidFromObject', 'objectId': held_obj['objectId']},
                      action_can_fail=action_can_fail)
    return True


#########################

def use_water_source_to_fill_or_clean_held(renv: RecordingEnv, water_source,
                                           skip_turning_on_water=False,
                                           empty_afterwards=False,
                                           action_can_fail=True):
    """
    Start with a held object. Go to the water source and put it in. Water will magically clean and fill everything in it


    :param renv:
    :param water_source: watersource
    :param skip_turning_on_water: Whether to fake out the model and NOT turn on the water
    :param empty_afterwards: Empty the held object afterwards
    :param action_can_fail:
    :return:
    """
    if water_source['objectType'] != 'Faucet':
        raise ValueError("needs 2 be faucet")

    sink_basins = sorted(env.all_objects_with_properties({'receptacle': True, 'objectType': 'SinkBasin'}),
                         key=lambda obj: env.position_dist(
                             water_source['position'], obj['position'], ignore_y=True)
                         )
    sink_basin = sink_basins[0]

    held_obj = renv.env.object_in_hand()
    if held_obj is None:
        if not action_can_fail:
            raise ValueError("No held obj")
        return False

    s_a = put_object_in_receptacle(renv, sink_basin['objectId'], navigate=True, action_can_fail=False)
    if not skip_turning_on_water:
        # Hack 1. clean everything under water
        for obj_id in env.get_object_by_id(sink_basin['objectId'])['receptacleObjectIds']:
            obj = env.get_object_by_id(obj_id)
            if obj['isDirty']:
                env.controller.step({'action': 'CleanObject', 'objectId': obj['objectId'], 'forceVisible': True})

            if not obj['isFilledWithLiquid']:
                env.controller.step({'action': 'FillObjectWithLiquid', 'objectId': obj['objectId'],
                                     'fillLiquid': 'water',
                                     'forceVisible': True})
        inch_closer_path(renv, water_source)

        # Now turn on the water
        s_a &= renv.step({'action': 'ToggleObjectOn', 'objectId': water_source['objectId']},
                         action_can_fail=action_can_fail)

        if random.random() < 0.8:
            s_a &= renv.step({'action': 'ToggleObjectOff', 'objectId': water_source['objectId']},
                             action_can_fail=action_can_fail)

    s_b = pickup_object(renv, held_obj['objectId'], navigate=False, action_can_fail=action_can_fail)

    if empty_afterwards:
        s_b &= pour_out_liquid(renv, action_can_fail=action_can_fail)
    return s_a and s_b


def slice_held_object(renv: RecordingEnv, anchor_pt=None, action_can_fail=False):
    """
    Slice the held object. but for this we need to put it DOWN first.
    :param renv:
    :param anchor_pt: put held object down somewhere NEAR here
    :return:
    """
    all_cutting_surfaces = renv.env.all_objects_with_properties({'objectType': 'CounterTop'})
    if len(all_cutting_surfaces) == 0:
        raise ValueError("No cutting surfaces")

    held_obj = env.object_in_hand()
    if held_obj is None:
        raise ValueError("No held obj")

    anchor_pt_loc = anchor_pt if anchor_pt is not None else renv.env.get_agent_location()
    all_cutting_surfaces = sorted(all_cutting_surfaces,
                                  key=lambda x: renv.env.position_dist(x['position'],
                                                                       anchor_pt_loc) + np.random.randn() * 0.25 + len(
                                      x['receptacleObjectIds']))
    surface = all_cutting_surfaces[0]
    put_object_in_receptacle(renv, surface['objectId'], navigate=True, action_can_fail=False)

    ok = renv.step({'action': 'SliceObject', 'objectId': held_obj['objectId']}, action_can_fail=action_can_fail)

    # Get constituent objects
    slices = sorted([x for x in env.visible_objects() if x['objectId'].startswith(held_obj['objectId'])],
                    key=lambda x: x['name'])
    random.shuffle(slices)
    if len(slices) == 0:
        raise ValueError("no slices?")

    renv.alias_object_id_to_old_object_id[slices[0]['objectId']] = held_obj['objectId']

    ok &= pickup_object(renv, slices[0]['objectId'], navigate=False, action_can_fail=action_can_fail)
    return ok


# 1. Put object X in receptacle Y, wait, take it out.

env = AI2ThorEnvironment(
    make_agents_visible=True,
    object_open_speed=0.05,
    restrict_to_initially_reachable_points=True,
    render_class_image=True,
    render_object_image=True,
    player_screen_height=384,
    player_screen_width=640,
    quality="Very High",
    x_display="0.{}".format(args.display),
    fov=90.0,
)


def _weighted_choice(weights):
    """
    Choose based on weights
    :param weights: they don't necessarily need to sum to 1
    :return:
    """
    weights_np = np.array(weights)
    assert np.all(weights_np >= 0)

    return int(np.random.choice(weights_np.size, p=weights_np / weights_np.sum()))

def _sample_bigger_receptacle_than_me(object_x, all_receptacles, min_ratio=2.0):
    """
    Sample a random object that's usually bigger than obj
    :param obj:
    :param receptacles: List of receptacles
    :return:
    """
    def _size3d(o):
        xyz = np.array([o['axisAlignedBoundingBox']['size'][k] for k in 'xyz'])
        return np.sqrt(xyz[0] * xyz[2])

    x_size3d = _size3d(object_x)+0.001
    r_weights = [(0.05 if _size3d(r)/x_size3d < min_ratio else 1.0) for r in all_receptacles]
    receptacle_y = all_receptacles[_weighted_choice(r_weights)]

    object_is_bigger = x_size3d > _size3d(receptacle_y)
    return receptacle_y, object_is_bigger

######################################

def _place_held_object_in_receptacle():
    """
    If we're ALREADY holding an object, place it in a receptacle
    :return:
    """
    all_receptacles = env.all_objects_with_properties({'receptacle': True})
    if any([x['objectType'] == 'Fridge' for x in all_receptacles]) and random.random() < 0.5:
        all_receptacles = env.all_objects_with_properties({'receptacle': True, 'objectType': 'Fridge'})
    receptacle_y = random.choice(all_receptacles)

    held_object = env.object_in_hand()
    if held_object is None:
        raise ValueError("No object?")

    renv = RecordingEnv(env,
                        text='Put the held $1 in $2.',
                        main_object_ids=(held_object['objectId'], receptacle_y['objectId']),
                        )
    put_object_in_receptacle(renv, receptacle_y['objectId'], navigate=True, action_can_fail=False)

    is_interesting = random.random() > 0.9
    return renv, is_interesting


def _put_object_x_in_receptacle_y():
    """
    Interesting interactions:

    * Put stuff into fridge

    :return:
    """
    if len(env.all_objects_with_properties({'pickupable': True})) == 0:
        raise ValueError('No pickupable objects')

    # Sample interesting things
    all_objects_x = env.all_objects_with_properties({'pickupable': True})
    x_weights = [2.0 if 'Food' in x['salientMaterials'] else 1.0 for x in all_objects_x]
    object_x = all_objects_x[_weighted_choice(x_weights)]

    ###############
    all_receptacles = env.all_objects_with_properties({'receptacle': True})
    if any([x['objectType'] == 'Fridge' for x in all_receptacles]) and random.random() < 0.5:
        all_receptacles = env.all_objects_with_properties({'receptacle': True, 'objectType': 'Fridge'})

    # Prioritize bigger receptacles
    receptacle_y, obj_bigger_than_receptacle = _sample_bigger_receptacle_than_me(object_x, all_receptacles)

    # Switch the toggle state too!
    new_toggle = None

    if receptacle_y['toggleable'] and random.random() < 0.5:

        new_toggle = 'Off' if receptacle_y['isToggled'] else 'On'
        text = 'Put $1 in $2, then turn $2 {}.'.format(new_toggle.lower())
    else:
        text = 'Put $1 in $2.'

    slice_first = object_x['sliceable'] and (not object_x['isSliced']) and random.random() < 0.5

    if slice_first:
        text = f'Slice $1. {text}'

    renv = RecordingEnv(env,
                        text=text,
                        main_object_ids=(object_x['objectId'], receptacle_y['objectId']),
                        )

    s_a = pickup_object(renv, object_x['objectId'], navigate=True, action_can_fail=False)
    if slice_first:
        slice_held_object(renv, anchor_pt=receptacle_y['position'], action_can_fail=False)
        object_x = env.object_in_hand()

    # Maybe we'll learn that big things can't be placed in small things
    # Here we're preemptively saying that the result will be 'interesting'
    can_keep = (random.random() > 0.9) and obj_bigger_than_receptacle
    s_b = put_object_in_receptacle(renv, receptacle_y['objectId'], navigate=True, action_can_fail=can_keep)
    if not s_b:
        return renv, True

    print("Put object in {} succeeds switching toggle state?".format(receptacle_y['objectId']), flush=True)

    # Switch toggle state
    if new_toggle is not None:
        renv.step({'action': f'ToggleObject{new_toggle}', 'objectId': object_x['objectId']})

    pickup_object(renv, object_x['objectId'], navigate=False, action_can_fail=False, force_visible=True)
    print("Pickup {} again succeeds".format(object_x['objectId']), flush=True)

    is_interesting = False
    # Check if any object changed temperature
    for oid, states in renv.object_id_to_states.items():
        temps = [x['ObjectTemperature'] for x in states.values()]
        if not all([x == temps[0] for x in temps]):
            is_interesting = True

    is_interesting |= (random.random() > 0.5)
    is_interesting |= slice_first
    is_interesting |= (new_toggle is not None)

    return renv, is_interesting


def _throw_object_x_at_y():
    """
    Interesting interactions:

    * If anything is breakable

    :return:
    """
    all_pickupable_objects_x = env.all_objects_with_properties({'pickupable': True})
    x_weights = [10.0 if (x['breakable'] or x['mass'] > 4.0) else 1.0 for x in all_pickupable_objects_x]
    if len(all_pickupable_objects_x) == 0:
        raise ValueError('No pickupable objects')

    all_objects_y = env.all_objects_with_properties({'pickupable': True})
    y_weights = [10.0 if (y['breakable'] and not y['pickupable']) else (
        4.0 if y['breakable'] else 1.0) for y in all_objects_y]

    object_x = all_pickupable_objects_x[_weighted_choice(x_weights)]
    object_y = all_objects_y[_weighted_choice(y_weights)]

    if object_x['objectId'] == object_y['objectId']:
        raise ValueError('objects are the same?')
    #####################

    hardness_options = {'softly': 10.0, 'normally': 100.0, 'aggressively': 1000.0}
    hardness = random.choice(sorted(hardness_options.keys()))

    renv = RecordingEnv(env,
                        text=f'Throw $1 at $2 {hardness}.',
                        main_object_ids=(object_x['objectId'], object_y['objectId'])
                        )

    s_a = pickup_object(renv, object_x['objectId'], navigate=True)
    print("Pickup {} succeeds".format(object_x['objectId']), flush=True)

    path2use = path_to_object(renv.env, object_y, angle_noise=0, dist_to_obj_penalty=0.1)
    while len(path2use) > 0 and path2use[-1]['action'].startswith(('Rotate', 'Look')):
        path2use.pop(-1)

    for p in path2use:
        renv.step(p)

    # Teleport, throw, then snap back to grid

    # Face object
    old_pos = renv.env.get_agent_location()
    new_pos = {k: v for k, v in old_pos.items()}

    new_pos['rotation'] = rotation_angle_to_object(object_y, renv.env.get_agent_location())
    new_pos['horizon'] = horizon_angle_to_object(object_y, renv.env.get_agent_location())
    renv.env.teleport_agent_to(**new_pos, ignore_y_diffs=True,
                               only_initially_reachable=False)

    if not renv.env.last_action_success:
        raise ValueError("teleport failed")

    if renv.env.get_agent_location()['y'] < -10:
        raise ValueError("negative coords")

    s_b = renv.step(dict(action='ThrowObject', moveMagnitude=hardness_options[hardness],
                         forceAction=True))

    # If something broke then things are interesting
    is_interesting = s_b and any([(x['isBroken'] or 'Cracked' in x['objectType']) for x in renv.new_items.values()])
    renv.env.teleport_agent_to(**old_pos, ignore_y_diffs=True)
    return renv, is_interesting


def _toggle_objectx():
    """
    Just turn on the object
    :return:
    """
    all_toggleable_objects = env.all_objects_with_properties({'toggleable': True})
    if len(all_toggleable_objects) == 0:
        raise ValueError("No toggleable objects")
    object_x = random.choice(all_toggleable_objects)

    is_on = object_x['isToggled']

    new_key = 'Off' if is_on else 'On'

    renv = RecordingEnv(env,
                        text='Turn $1 {}.'.format(new_key.lower()),
                        main_object_ids=(object_x['objectId']), )

    for p in path_to_object(renv.env, target_object=object_x):
        renv.step(p)
    s = renv.step({'action': 'ToggleObject{}'.format(new_key), 'objectId': object_x['objectId']})

    return renv, s


def _simple_change(action_name, property_name, affordance_name):
    """
    Perform a simple change -- apply action `action_name` to objects with the affordance `affordance_name`

    :param action_name:
    :param property_name:
    :param affordance_name:
    :return:
    """
    all_objects = env.all_objects_with_properties({affordance_name: True, property_name: False})
    if len(all_objects) == 0:
        raise ValueError("No valid objects for {}".format(action_name))
    object_x = random.choice(all_objects)

    renv = RecordingEnv(env,
                        text='{} $1.'.format(action_name.split('Object')[0].capitalize()),
                        main_object_ids=(object_x['objectId'],)
                        )

    for p in path_to_object(renv.env, target_object=object_x):
        renv.step(p)
    s = renv.step({'action': action_name, 'objectId': object_x['objectId']})
    if object_x['pickupable']:
        pickup_object(renv, object_x['objectId'], navigate=False, action_can_fail=True)
    return renv, s

def _dirty_objectx():
    return _simple_change('DirtyObject', 'isDirty', 'dirtyable')


def _slice_objectx():
    sliceables = env.all_objects_with_properties({'sliceable': True, 'isSliced': False})

    if (len(sliceables) == 0):
        raise ValueError("cant slice")
    sliceable = random.choice(sliceables)

    renv = RecordingEnv(env, text='Slice $1.',
                        main_object_ids=tuple([sliceable['objectId']]))

    pickup_object(renv, sliceable['objectId'], navigate=True, action_can_fail=False)
    slice_held_object(renv, anchor_pt=env.get_agent_location(), action_can_fail=False)
    return renv, True

def _sink_clean_and_put_away_object():
    """
    Clean an object and put it in a cabinet. We'll make an object dirty if need be
    :return:
    """
    cabinets_y = [x for x in env.all_objects_with_properties({'receptacle': True}) if
                  x['objectType'] in ('Cabinet', 'Drawer', 'Shelf', 'CounterTop', 'StoveBurner')]
    if len(cabinets_y) == 0:
        raise ValueError("No cabinets")

    all_objects = env.all_objects_with_properties({'dirtyable': True, 'pickupable': True})
    if len(all_objects) == 0:
        raise ValueError("no dirtyable objects")

    all_faucets = env.all_objects_with_properties({'objectType': 'Faucet'})
    if len(all_faucets) == 0:
        raise ValueError("No faucets")

    random.shuffle(all_objects)

    object_types_to_dirty = set([obj['objectType'] for obj in all_objects[:5]])
    for ot in object_types_to_dirty:
        env.controller.step(action='SetObjectStates', SetObjectStates={
            'objectType': ot,
            'stateChange': 'dirtyable',
            'isDirty': True,
        })

    all_objects = env.all_objects_with_properties({'dirtyable': True, 'isDirty': True, 'pickupable': True})
    if len(all_objects) == 0:
        raise ValueError("no dirtyable objects V2")

    object_x = random.choice(all_objects)
    cabinet_y, _ = _sample_bigger_receptacle_than_me(object_x, cabinets_y, min_ratio=4.0)

    faucets = sorted(all_faucets, key=lambda obj: env.position_dist(
        object_x['position'], obj['position']) + np.random.randn() * 0.25
                     )
    if len(faucets) == 0:
        raise ValueError("No faucets")
    faucet = faucets[0]


    # Receptacles near faucet
    sink_basins = sorted(env.all_objects_with_properties({'receptacle': True, 'objectType': 'SinkBasin'}),
                         key=lambda obj: env.position_dist(
                             faucet['position'], obj['position'])
                         )
    if len(sink_basins) == 0:
        raise ValueError("No sinkbasin")

    sink_basin = sink_basins[0]

    renv = RecordingEnv(env,
                        text='Clean $1 in $2 using $3, then try to put it in $4.',
                        main_object_ids=(object_x['objectId'], sink_basin['objectId'], faucet['objectId'],
                                         cabinet_y['objectId']), )

    pickup_object(renv, object_x['objectId'], navigate=True, action_can_fail=False)
    is_interesting = use_water_source_to_fill_or_clean_held(renv, water_source=faucet,
                                                            skip_turning_on_water=random.random() > 0.9,
                                                            empty_afterwards=random.random() < 0.9)
    if not is_interesting:
        return renv, False
    s_e = put_object_in_receptacle(renv, cabinet_y['objectId'], navigate=True, action_can_fail=True)
    return renv, (s_e or random.random() > 0.8)


def _toast_bread():
    """
    Toast bread in the toaster.
    :return:
    """

    toasters = env.all_objects_with_properties({'objectType': 'Toaster'})
    breads = [x for x in env.all_objects() if x['objectType'] in ('BreadSliced', 'Bread')]

    if (len(toasters) == 0) or len(breads) == 0:
        raise ValueError("No bread")
    bread = random.choice(breads)
    toaster = random.choice(toasters)

    renv = RecordingEnv(env,
                        text='Slice $1 if needed, toast it, then put it in $2.',
                        main_object_ids=(bread['objectId'], toaster['objectId']))

    pickup_object(renv, bread['objectId'], navigate=True, action_can_fail=False)

    if (bread['objectType'] == 'Bread') and bread['sliceable'] and (not bread['isSliced']):  # so not sliced
        slice_held_object(renv, anchor_pt=toaster['position'], action_can_fail=False)

    put_object_in_receptacle(renv, toaster['objectId'], navigate=True, action_can_fail=False)

    toaster = env.get_object_by_id(toaster['objectId'])
    if len(toaster['receptacleObjectIds']) == 0:
        raise ValueError("No items in toaster")

    inch_closer_path(renv, toaster)
    if random.random() < 0.9:
        renv.step({'action': 'ToggleObjectOn', 'objectId': toaster['objectId']}, action_can_fail=False)
        if random.random() < 0.6:
            renv.step({'action': 'ToggleObjectOff', 'objectId': toaster['objectId']}, action_can_fail=False)

    bread = env.get_object_by_id(toaster['receptacleObjectIds'][0])
    s_c = pickup_object(renv, bread['objectId'], navigate=False, action_can_fail=False)

    # Just try to put it away somewhere?
    plates = [x for x in env.all_objects_with_properties({'receptacle': True}) if
                  x['objectType'] in ('Drawer', 'Shelf', 'CounterTop', 'Plate', 'GarbageCan')]
    if len(plates) == 0:
        return renv, True
    s_e = put_object_in_receptacle(renv, random.choice(plates)['objectId'], navigate=True, action_can_fail=True)
    return renv, s_c


def _brew_coffee():
    """
    Brews coffee using a coffee machine
    :return:
    """
    mugs = env.all_objects_with_properties({'objectType': 'Mug'})
    coffee_machines = env.all_objects_with_properties({'objectType': 'CoffeeMachine'})

    if len(mugs) == 0 or len(coffee_machines) == 0:
        raise ValueError("no coffee stuff")

    cmachine = random.choice(coffee_machines)
    mug = sorted(mugs, key=lambda x: env.position_dist(x['position'],
                                                       cmachine['position']) + 0.25 * np.random.randn())[0]
    env.controller.step(action='EmptyLiquidFromObject', objectId=mug['objectId'])

    if cmachine['isToggled']:
        env.controller.step(action='ToggleObjectOff', objectId=cmachine['objectId'])

    renv = RecordingEnv(env,
                        text='Try to brew coffee in $1 using $2.',
                        main_object_ids=(mug['objectId'], cmachine['objectId'],)
                        )
    pickup_object(renv, mug['objectId'], navigate=True, action_can_fail=False)
    put_object_in_receptacle(renv, cmachine['objectId'], navigate=True, action_can_fail=False)

    # Fake out the model!
    if random.random() < 0.9:
        renv.step({'action': 'ToggleObjectOn', 'objectId': cmachine['objectId']}, action_can_fail=False)
        if random.random() < 0.6:
            renv.step({'action': 'ToggleObjectOff', 'objectId': cmachine['objectId']}, action_can_fail=False)

    pickup_object(renv, mug['objectId'], navigate=False, action_can_fail=False)
    pour_out_liquid(renv, action_can_fail=False)

    is_interesting = True
    return renv, is_interesting


def _slice_and_fry():
    """
    Slice potato / etc. and then cook it
    :return:
    """
    sliceables = env.all_objects_with_properties({'sliceable': True, 'isSliced': False})

    # Only these objects can be cooked
    sliceables = [x for x in sliceables if x['objectType'] in ('Bread', 'EggCracked', 'Potato', 'Egg',
                                                               'Apple', 'AppleSliced', 'PotatoSliced', 'Tomato', 'TomatoSliced')]

    pans = [x for x in env.all_objects() if x['objectType'] in ['Pan', 'Pot']]

    knobs_and_burners = group_knobs_and_burners(env)
    if (len(sliceables) == 0) or len(knobs_and_burners) == 0 or len(pans) == 0:
        raise ValueError("cant slicie and cook")

    pan = random.choice(pans)

    # Check if we have a burner already
    burner = None
    knob = None
    blist = [y[1]['objectId'] for y in knobs_and_burners]
    for x in pan['parentReceptacles']:
        if x in blist:
            ind = blist.index(x)
            knob, burner = knobs_and_burners[ind]

    pan_already_on_stove = burner is not None

    # Delete all objects on stove and pick a burner/knob
    if not pan_already_on_stove:
        # for sb in env.all_objects_with_properties({'objectType': 'StoveBurner'}):
        #     for oid in sb['receptacleObjectIds']:
        #         env.controller.step(action='RemoveFromScene', objectId=oid)
        knob, burner = random.choice(knobs_and_burners)

    sliceable = random.choice(sliceables)
    do_slice = random.random() < 0.5
    if do_slice:
        text = f'Slice $1, put it in $2, then try to fry it on $3 using $4. Last, pick it up.'
    else:
        text = f'Put $1 on $2, then try to fry it on $3 using $4. Last, pick it up.'

    renv = RecordingEnv(env, text=text,
                        main_object_ids=tuple([sliceable['objectId'], pan['objectId'],
                                               burner['objectId'], knob['objectId']]))

    pickup_object(renv, sliceable['objectId'], navigate=True, action_can_fail=False)

    if sliceable['objectType'] != 'Egg':
        # Slice the object near something
        slice_held_object(renv, anchor_pt=pan['position'], action_can_fail=False)
        put_object_in_receptacle(renv, pan['objectId'], navigate=True, action_can_fail=False)
    else:
        put_object_in_receptacle(renv, pan['objectId'], navigate=True, action_can_fail=False)
        renv.step({'action': 'SliceObject',
                   'objectId': sliceable['objectId']}, action_can_fail=False)
        pan_rids = env.get_object_by_id(pan['objectId'])['receptacleObjectIds']
        if len(pan_rids) == 1:
            if pan_rids[0] != sliceable['objectId']:
                renv.alias_object_id_to_old_object_id[pan_rids[0]] = sliceable['objectId']

    if not pan_already_on_stove:
        pickup_object(renv, pan['objectId'], navigate=True, action_can_fail=False)
        put_object_in_receptacle(renv, burner['objectId'], navigate=True, action_can_fail=False)

    # Try to turn on the stove
    for p in path_to_object(env=renv.env, target_object=knob, dist_to_obj_penalty=2.0):
        renv.step(p, action_can_fail=False)
    renv.step({'action': 'ToggleObjectOn',
               'objectId': knob['objectId']}, action_can_fail=False)

    env.step({'action': 'Pass'})
    env.step({'action': 'Pass'})
    env.step({'action': 'Pass'})
    env.step({'action': 'Pass'})

    if random.random() < 0.5:
        renv.step({'action': 'ToggleObjectOff',
                   'objectId': knob['objectId']}, action_can_fail=False)

    inch_closer_path(renv=renv, target_receptacle=pan)

    # Pickup object, it might be different now
    objects_in_pan = env.get_object_by_id(pan['objectId'])['receptacleObjectIds']
    if len(objects_in_pan) != 1:
        raise ValueError("invalid # of objects in pan")

    renv.alias_object_id_to_old_object_id[objects_in_pan[0]] = sliceable['objectId']
    pickup_object(renv, objects_in_pan[0], navigate=False, action_can_fail=False, force_visible=True)

    return renv, True


def _cook_object_in_microwave():
    """
    Cook object in microwave - cup of water OR potato OR egg.
    :return:
    """
    all_pickupable_objects_x = env.all_objects_with_properties({'pickupable': True})
    microwaves = env.all_objects_with_properties({'objectType': 'Microwave'})
    if len(microwaves) == 0 or len(all_pickupable_objects_x) == 0:
        raise ValueError("Can't cook obj in microwave")

    if random.random() < 0.5:
        fill_all_objects_with_liquid(env)


    x_weights = [10.0 if (x['cookable'] or x['isFilledWithLiquid']) else 1.0 for x in all_pickupable_objects_x]

    obj_id = all_pickupable_objects_x[_weighted_choice(x_weights)]['objectId']
    microwave_id = random.choice(microwaves)['objectId']

    renv = RecordingEnv(env,
                        text='Put $1 in $2. Try to turn on $2, then take $1 out.',
                        main_object_ids=(obj_id, microwave_id))

    obj = env.get_object_by_id(obj_id)
    do_slice = obj['sliceable'] and (not obj['isSliced']) and (random.random() < 0.5)
    pickup_object(renv, obj_id, navigate=True, action_can_fail=False)
    if do_slice:
        slice_held_object(renv, anchor_pt=env.get_object_by_id(microwave_id)['position'], action_can_fail=False)

    obj_id2 = renv.env.object_in_hand()['objectId']

    # Check objects already in microwave
    objects_already_in_microwave = env.get_object_by_id(microwave_id)['receptacleObjectIds']


    put_object_in_receptacle(renv, microwave_id, navigate=True, action_can_fail=False)
    # We'll say it's always interesting because we fail if obj can't go in the microwave

    # Turn on the microwave
    if random.random() < 0.9:
        renv.step({'action': 'ToggleObjectOn', 'objectId': microwave_id})
        if random.random () < 0.9:
            renv.step({'action': 'ToggleObjectOff', 'objectId': microwave_id})

    microwave = env.get_object_by_id(microwave_id)

    microwave_objs = sorted(microwave['receptacleObjectIds'], key=lambda x: random.random() + float(x in objects_already_in_microwave))
    obj_id3 = microwave_objs[0]

    renv.alias_object_id_to_old_object_id[obj_id2] = obj_id
    renv.alias_object_id_to_old_object_id[obj_id3] = obj_id
    pickup_object(renv, obj_id3, navigate=False, action_can_fail=False)
    return renv, True


def _fill_objectx_with_liquid():
    """
    Fill objectX with liquid and then empty it.
    :return:
    """

    canfill = env.all_objects_with_properties({'canFillWithLiquid': True, 'pickupable': True})
    water_source = env.all_objects_with_properties({'objectType': 'Faucet'})

    if len(canfill) == 0 or len(water_source) == 0:
        raise ValueError("cant fill object")

    canfill = random.choice(canfill)
    water_source = random.choice(water_source)

    renv = RecordingEnv(env,
                        text='Fill $1 with liquid using $2, then empty it.',
                        main_object_ids=(canfill['objectId'], water_source['objectId']))

    pickup_object(renv, canfill['objectId'], navigate=True, action_can_fail=False)
    is_interesting = use_water_source_to_fill_or_clean_held(renv=renv, water_source=water_source, action_can_fail=False,
                                                            empty_afterwards=True)
    return renv, is_interesting

def _random_string(n):
    pool = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    candidates = np.random.choice(len(pool), size=n, replace=True)
    return ''.join([pool[c] for c in candidates])


random.seed(args.seed)
np.random.seed(args.seed)

scene_nums = sorted([int(x.split('_physics')[0].split('FloorPlan')[1]) for x in env.controller.scenes_in_build if x.endswith('_physics')])
scenes = pd.DataFrame(scene_nums, columns=['id'])
room_types = {0: 'Kitchens', 2: 'livingRooms', 3: 'bedrooms', 4: 'bathrooms', 5: 'foyers'}
scenes['rtype'] = [room_types[x // 100] for x in scene_nums]
scenes['weight'] = 1.0 #+ 9.0 * (scenes['rtype'] == 'Kitchens')

# I counted how often things occur naturally so I can weight inversely to it
###################
tasks_to_expected_count = [
    (_put_object_x_in_receptacle_y, 55),
    (_throw_object_x_at_y, 202),
    (_toggle_objectx, 741),
    (_slice_objectx, 400),
    (_dirty_objectx, 564),
    (_sink_clean_and_put_away_object, 71),
    (_toast_bread, 136),
    (_brew_coffee, 376),
    (_slice_and_fry, 81),
    (_cook_object_in_microwave, 76),
    (_fill_objectx_with_liquid, 98),
]

# If it exists, use all existing things as a "prior"
prior_counts = np.zeros(len(tasks_to_expected_count), dtype=np.int64)
name2ind = {x[0].__name__.strip('_'): i for i, x in enumerate(tasks_to_expected_count)}
for fn in glob.iglob(os.path.join(args.out_path, '*/*.h5')):
    task = fn.split('/')[-2]
    prior_counts[name2ind[task]] += 1

# 0.1 is expected odds of success? just made that up
prior_p_success = prior_counts.astype(np.float32) / ((np.sum(prior_counts)+1.0) / 0.1) * prior_counts.size
prior_counts = np.stack([prior_counts, (prior_counts * (1 - prior_p_success) / (prior_p_success+1e-8)).astype(np.int64)])

if prior_counts.min() == 0:
    denom = 10000
    denom_adj = 1000.0
    prior_counts = np.array([x[1] for x in tasks_to_expected_count], dtype=np.int64) / denom_adj
    prior_p_success = prior_counts.astype(np.float32) / (denom / prior_counts.size) * denom_adj
    prior_counts = np.stack([prior_counts, (prior_counts * (1 - prior_p_success) / prior_p_success)])
    prior_counts += 1.0
    print(prior_counts, flush=True)

counts_now = np.zeros_like(prior_counts)

def generate_loop():
    must_reset = True # This gets overwritten later
    while counts_now.sum() < args.nex:
        if must_reset or (random.random() > 0.5):
            sind = _weighted_choice(scenes['weight'])
            env.reset(scene_name='FloorPlan{}_physics'.format(scene_nums[sind]))
            env.controller.step(action='InitialRandomSpawn', randomSeed=random.randint(0, 2**32),
                                forceVisible=random.choice([False, True]), numPlacementAttempts=3,
                                excludedReceptacles=['Toaster', 'SinkBasin', 'Sink'],
                                placeStationary=True)
            must_reset = False

        # Randomize, let's say 50% chance SOMETHING will happen which is 1-(1-pr)^7 = 0.5
        randomize_toggle_states(env, random_prob=0.91)
        # Close all receptacles since having them open is annoying
        close_everything(env)

        env.randomize_agent_location()
        held_object = env.all_objects_with_properties({'isPickedUp': True})
        if len(held_object) > 0:
            must_reset = True
            continue

        env.recompute_reachable()

        # Weight everything by its probability of success
        p_success = (counts_now + prior_counts)[0] / (counts_now + prior_counts).sum(0).astype(np.float32)
        ind = _weighted_choice(1.0/p_success)
        try:
            task_fn, _ = tasks_to_expected_count[ind]
            renv, is_interesting = tasks_to_expected_count[ind][0]()
            if is_interesting:
                counts_now[0, ind] += 1
                renv.meta_info['task_name'] = task_fn.__name__
                yield renv
            else:
                counts_now[1, ind] += 1

        except ValueError as e:
            print(str(e))
            continue

if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)

for x in tqdm(generate_loop()):
    # Save it.
    task_folder = os.path.join(args.out_path, x.meta_info['task_name'].strip('_'))
    if not os.path.exists(task_folder):
        os.mkdir(task_folder)

    x.save(os.path.join(task_folder, '{}.h5'.format( _random_string(12))))