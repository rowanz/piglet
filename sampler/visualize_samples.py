"""
This is used for visualizing -- not really needed otherwise

"""
import os
os.environ["DISPLAY"] = ""
import argparse
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import pandas as pd

from matplotlib.animation import FuncAnimation

COLORS = np.array(
    [0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188,
     0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
     1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
     0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
     0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500,
     0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
     0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500,
     1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
     0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
     0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
     1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
     0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
     0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
     0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
     0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429, 0.429,
     0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
     ]
).astype(np.float32).reshape((-1, 3))

parser = argparse.ArgumentParser(description='Visualize Trajectories')
parser.add_argument(
    '-fn',
    dest='fn',
    type=str,
    help='fn to use'
)
args = parser.parse_args()
data_h5 = h5py.File(args.fn, 'r')

meta_info = json.loads(data_h5['meta_info'][()].decode('utf-8'))
meta_info['task_name'] = meta_info['task_name'].strip('_')
output_actions = json.loads(data_h5['output_actions'][()].decode('utf-8'))
output_actions.append({'action': 'Done'})
aliases = json.loads(data_h5['alias_object_id_to_old_object_id'][()].decode('utf-8'))
object_id_to_states = json.loads(data_h5['object_id_to_states'][()].decode('utf-8'))

# Last action / next action
def action_txt(action_t):
    action_str = [action_t['action']]
    if 'objectId' in action_t:
        action_str.append(action_t['objectId'].split('|')[0])
    if 'receptacleObjectId' in action_t:
        action_str.append(action_t['receptacleObjectId'].split('|')[0])

    return '< {} >'.format(' , '.join(action_str))
action_txt = [action_txt(action_t) for action_t in output_actions]

goal_txt = '{}:\n{}'.format(meta_info['task_name'], meta_info['text'])
for i, mid in enumerate(meta_info['main_object_ids']):
    goal_txt = goal_txt.replace(f'${i+1}', '[{}]'.format(mid.split('|')[0]))
# Cheap latex style alignment
goal_txt = goal_txt.replace('. ', '.\n')
for s2 in goal_txt.split('\n'):
    if len(s2) < 30:
        continue
    for s1 in s2.split(', ')[:-1]:
        goal_txt = goal_txt.replace(s1 + ', ', s1 + ',\n')



ims = []

num_frames = data_h5['frames'].shape[0]
frames = np.array(data_h5['frames'], dtype=np.int32)
IM_SIZE = (384, 640)
DPI = 128

fig = plt.figure(frameon=False)
fig.set_size_inches(IM_SIZE[1] / DPI, IM_SIZE[0] / DPI)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.axis('off')
fig.add_axes(ax)

# Compute interesting state changes accross aliases
def _column_interestingness_score(col):
    vcs = col.value_counts().values.astype(np.float32)
    vcs_p = vcs / vcs.sum()
    return -np.sum(vcs_p * np.log(vcs_p))


dfs = {}
for oid in meta_info['main_object_ids']:
    df = []
    oid_and_aliases = [oid] + [k for k, v in aliases.items() if v == oid]
    for t in range(num_frames):
        # Add a dummy None which will get overwritten
        df.append(None)
        for oid_k in oid_and_aliases:
            if str(t) in object_id_to_states[oid_k]:
                df[t] = object_id_to_states[oid_k][str(t)]
        if df[t] is None:
            assert t > 0
            df[t] = df[t-1]
    df = pd.DataFrame(df)

    # These things don't change or are not interesting
    for k in ['name', 'objectType', 'objectId', 'visible', 'isMoving', 'distance']:
        del df[k]
    # Clean IDs
    df['parentReceptacles'] = df['parentReceptacles'].apply(lambda x: '&'.join([y.split('|')[0] for y in x]))
    df['receptacleObjectIds'] = df['receptacleObjectIds'].apply(lambda x: '&'.join([y.split('|')[0] for y in x]))
    df['salientMaterials'] = df['salientMaterials'].apply(lambda x: '&'.join(x))

    # Now order by the interestingness of the columns
    columns_scored = [(col,  _column_interestingness_score(df[col])) for col in df.columns]
    columns_scored = sorted(columns_scored, key=lambda x: -x[1])
    df = df[[c[0] for c in columns_scored if c[1] > 0.0]]
    dfs[oid] = df


def update(t):
    ax.clear()
    ax.imshow(frames[t], vmin=0, vmax=255)

    # Next action
    lastlast_action_str = action_txt[t-2] if t > 1 else ''
    last_action_str = action_txt[t-1] if t > 0 else ''
    next_action_str = action_txt[t] if t < len(output_actions) else '< Done >'

    ax.text(
        20, 24,
        f'{lastlast_action_str}',
        fontsize=8,
        family='serif',
        bbox=dict(
            facecolor=COLORS[0],
            alpha=0.4, pad=0, edgecolor=COLORS[0]),
        color='white',
        alpha=0.25)
    ax.text(
        20, 40,
        f'{last_action_str}',
        fontsize=8,
        family='serif',
        bbox=dict(
            facecolor=COLORS[0],
            alpha=0.4, pad=0, edgecolor=COLORS[0]),
        color='white',
        alpha=0.5)
    ax.text(
        20, 56,
        f'{next_action_str}',
        fontsize=8,
        family='serif',
        bbox=dict(
            facecolor=COLORS[0],
            alpha=0.4, pad=0, edgecolor=COLORS[0]),
        color='white')
    ax.text(
        IM_SIZE[1]-10, 40,
        goal_txt,
        fontsize=6,
        family='serif',
        bbox=dict(
            facecolor=COLORS[1],
            alpha=0.4, pad=0, edgecolor=COLORS[1]),
        color='white',
        ha='right', ma='left', va='top')

    # Put text in bottom half
    for i, oid in enumerate(meta_info['main_object_ids']):
        status_text = ',   '.join(['{}={}'.format(x[0], x[1] if not isinstance(x[1], float) else '{:.2f}'.format(x[1]))
                                   for x in dfs[oid].iloc[t].items() if len(str(x[1])) > 0][:3])
        ax.text(
            20, IM_SIZE[0] - 16*len(meta_info['main_object_ids']) + 16*i,
            '{}: {}'.format(oid.split('|')[0], status_text),
            fontsize=8,
            family='serif',
            bbox=dict(
                facecolor=COLORS[2+i],
                alpha=0.4, pad=0, edgecolor=COLORS[2+i]),
            color='white',
            ha='left')


ani = FuncAnimation(fig, update, frames=num_frames)
ani.save('{}-{}.mp4'.format(meta_info['task_name'], args.fn.split('/')[-1].split('.')[0]), writer='ffmpeg', dpi=DPI, fps=1, extra_args=['-vcodec', 'libx264'])