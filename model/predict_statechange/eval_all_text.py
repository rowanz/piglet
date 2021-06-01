"""
This has stuff for evaluating generation
"""
import sys

sys.path.append('../../')
import os
import numpy as np

from model.lm.modeling import GroverModel, LMConfig, create_initializer, _top_p_sample
from model.model_utils import dropout, get_assignment_map_from_checkpoint, construct_host_call

from model.neat_config import NeatConfig
from model import optimization
import tensorflow as tf
import json

from model.model_utils import pad_to_fixed_size as p2fsz
from model.model_utils import get_shape_list, switch_block
import pandas as pd
from data.zeroshot_lm_setup.encoder import get_encoder, Encoder
from tfrecord.tfrecord_utils import S3TFRecordWriter, int64_list_feature, int64_feature
from data.concept_utils import get_concepts_from_bpe_encoded_text, concept_df, mapping_dict, get_glove_embeds
from model.predict_statechange.finetune_dataloader import get_statechange_dataset, create_tfrecord_statechange, \
    evaluate_statechange
from model.interact.dataloader import input_fn_builder
from model.interact.modeling import StateChangePredictModel, embed_with_embedding_table
from data.thor_constants import numpy_to_instance_states
from model.transformer import residual_mlp
# from nltk.translate.bleu_score import sentence_bleu
# from allennlp.common.util import get_spacy_model
# spacy = get_spacy_model('en_core_web_sm', pos_tags=False, parse=False, ner=False)
# def _tokenize(s):
#     return [x.orth_.lower() for x in spacy(s.replace('\n\n', ' ')) if not x.is_punct]
import sacrebleu
import bert_score


ref_keys = ['annot', 'extra_annot0']
hyp_model = 'extra_annot1'


all_dsets = get_statechange_dataset()
def get_eval_for_a_split(split, is_zs=None):
    """
    Gets the evaluation stuff for a single SPLIT.
    :param split:
    :param is_zs:
    :return:
    """
    assert split in ('val', 'test')
    dset = all_dsets[split]

    if is_zs is None:
        to_include = [True for item in dset]
    elif is_zs == True:
        to_include = [item['is_zs'] for item in dset]
    else:
        to_include = [not item['is_zs'] for item in dset]

    models = {
        'baseline': 'PUT_PREDICTIONS_HERE_.CSV',
        'ours': f'gs://ipk-europe-west4/models/jan23/predictsc~hsize=256~act=tanh~lr=5e-6~ne=80~generate=True~train_time_symbolic_actions_prob=1.0~lm_loss_coef=1.0~version=v47~freeze=True/{split}_gens.csv',
        'zslm': f'gs://ipk-europe-west4/models/jan23/predictsc~hsize=256~act=tanh~lr=5e-6~ne=80~generate=True~train_time_symbolic_actions_prob=1.0~lm_loss_coef=1.0~version=v47~zslm=True/{split}_gens.csv',
    }

    all_preds = {}

    for model, fn in models.items():
        with tf.gfile.Open(fn, 'r') as f:
            all_preds[model] = pd.read_csv(f)['post_pred'].tolist()
            if any([not isinstance(x, str) for x in all_preds[model]]):
                nn = len([x for x in all_preds[model] if not isinstance(x, str)])
                print(f"NANs with {model}: {nn}")
                all_preds[model] = [x if isinstance(x, str) else dset[i]['annot']['postcondition_language'] for i, x in enumerate(all_preds[model])]

    all_preds = {k: [v2 for v2, ti in zip(v, to_include) if ti] for k, v in all_preds.items()}
    all_preds['human'] = [x[hyp_model]['postcondition_language'] for x in dset]


    dset_filtered = [x for x, ti in zip(dset, to_include) if ti]
    refs = [tuple([x[k]['postcondition_language'] for k in ref_keys]) for x in dset_filtered]
    refs2 = [[x[k]['postcondition_language'] for x in dset_filtered] for k in ref_keys]


    df = []
    # Columns are Human Eval (TBD), BLEU, and BERTScore
    # Use 100ex per model
    for model, res in all_preds.items():
        item = {
            'model': model,
            'Human': 0.0,
        }
        bleu_score = sacrebleu.corpus_bleu(sys_stream=res, ref_streams=refs2, lowercase=True)
        item['BLEU'] = bleu_score.score
        P, R, F1 = bert_score.score(res, refs, verbose=False, model_type='bert-large-uncased')
        item['BERTScore'] = float(F1.mean())
        item['nex'] = len(res)
        df.append(item)
    return pd.DataFrame(df).set_index('model', drop=True)

df_val = get_eval_for_a_split('val')
df_test = get_eval_for_a_split('test', is_zs=None)
# df_testzs = get_eval_for_a_split('test', is_zs=True)

out_txt = 'Model & \multicolumn{2}{c}{BLEU} & \multicolumn{2}{c}{BERTScore} & Human Eval'
out_txt += '\\\\ \n'
out_txt += '     & Val & Test               & Val & Test                    & Test'
for model in ['t5small', 'baseline', 'ours', 'human']:
    out_txt += '\\\\ \n'

    res = [df_val.loc[model, 'BLEU'], df_test.loc[model, 'BLEU'],
           100 * df_val.loc[model, 'BERTScore'], 100 * df_test.loc[model, 'BERTScore'],
           df_val.loc[model, 'Human'], df_test.loc[model, 'Human'],
           ]

    res_txt = [model] + ['{:.1f}'.format(x) for x in res]
    out_txt += '&'.join(['{:>10s}'.format(a) for a in res_txt])
































#
#
#
#
#
#
#
#
# for model, res in all_preds.items():
#
#     print(f"BLEU {model}: {bleu_score}")
#     P, R, F1 = bert_score.score(res, refs, lang="en", verbose=False)
#     f1_mean = float(F1.mean())
#     print(f"BERTSSORE {model}: {f1_mean:.3f}")
#
#
# for i in range(100):
#     refs_i = ' \n -> '.join(refs[i])
#     ap = '\n'.join(sorted([f'{k}: {v[i]}' for k, v in all_preds.items()]))
#     print(f"EX {i} {refs_i}\n{ap}\n~~~~")