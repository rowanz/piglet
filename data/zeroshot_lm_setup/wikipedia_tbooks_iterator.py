"""
Essentially what we want to do is get MAXLEN chunks from the datasets

A few things we want to do
* Skip sentences that have banned words
"""
import sys

sys.path.append('../../')
import nlp
import glob
from tqdm import tqdm, trange
import ftfy
import regex as re
import argparse
import random
from tfrecord.tfrecord_utils import S3TFRecordWriter
import pandas as pd
from data.zeroshot_lm_setup.encoder import get_encoder, random_sliding_window

wikipedia = nlp.load_dataset('wikipedia', '20200501.en')['train']
tbooks_fns = sorted(glob.glob('/home/rowan/datasets3/tbooks/*/*.txt'))
tbooks_fns = [x for x in tbooks_fns if not x.endswith('-all.txt')]

TOTAL_LEN = len(wikipedia) + len(tbooks_fns)


def _open_all_encodings(fn, encodings=('utf-8', 'latin1')):
    if len(encodings) == 0:
        return None
    try:
        with open(fn, 'r', encoding=encodings[0]) as f:
            stream = ftfy.fix_file(f, encoding=encodings[0])
            doc = ''.join([x for x in stream]).strip()
            return doc
    except UnicodeDecodeError as e:
        print("Error {} with encoding {} on {} -> {}".format(str(e), fn, encodings[0], encodings[1:]), flush=True)
        return _open_all_encodings(fn, encodings[1:])


def _skip_extra_wikipedia_stuff(x):
    x_skipped = \
        re.split(
            r'\n+\s*(see also|external links|references|further reading|notes|"type": "FeatureCollection",|{{)\s*\n+',
            x, flags=re.IGNORECASE)[0]
    if len(x_skipped) == 0:
        # print(f"??? {x}", flush=True)
        return None
    x_skipped2 = ftfy.fix_text(x_skipped.strip())
    return x_skipped2


def _get_document(i):
    if i >= TOTAL_LEN:
        print(f"Invalid index {i}: wikipedia={len(wikipedia)} tbooks={len(tbooks_fns)}", flush=True)
    if i < len(tbooks_fns):
        return _open_all_encodings(tbooks_fns[i])
    else:
        return _skip_extra_wikipedia_stuff(wikipedia[i - len(tbooks_fns)]['text'])


df = pd.read_csv('../categories_and_banned_words.tsv', delimiter='\t')
df = df[df['is_zeroshot']]

banned_words = set()
for _, row in df.iterrows():
    for col in ['thor_name', 'coco_name', 'vg_name', 'oi_name', 'preliminary_banned_terms']:
        if not pd.isnull(row[col]):
            for w in row[col].split(','):
                banned_words.add(w.lower())
                banned_words.add(w.lower().replace(' ', ''))
bw_regex = re.compile(r'\b(' + r'|'.join(banned_words) + r')\b', flags=re.IGNORECASE)


#####################
def skip_all_sentences_with_banned_words(doc):
    """
    Given a document (aka a string), we need to iterate through its sentences and filter out any sentence that has a bad word.

    This can result in multiple documents.
    :param doc: string
    :return: list of strings
    """
    # https://stackoverflow.com/questions/44244583/splitting-on-regex-without-removing-delimiters
    # This cuts out the last sentence unless you do 'doc + '\n' but actually that's probably OK
    d2 = re.findall('.*?[.!\?\n]+', doc)

    new_l = []
    for l in d2:
        bad_match = bw_regex.findall(l)
        if len(bad_match) > 0:
            # print(f"SKIP {bad_match}: {l}", flush=True)
            rv = ''.join(new_l).strip('\n')
            if len(rv) > 8:
                yield rv
            new_l = []
        else:
            new_l.append(l)
    rv = ''.join(new_l).strip('\n') + '\n'  # Anything ending with \n means it's at the end of the document
    if len(rv) > 8:
        yield rv


def document_iterator(inds, min_doc_character_len=64, noskip=False):
    """
    Go through the documents, skipping any with banned words
    :param inds: All from 0 < TOTAL_LEN
    :param min_doc_character_len: Skip the document if it's less than this # of characters
    :param noskip: whether to actually skip banned words
    :return:
    """
    for i in tqdm(inds):
        doc = _get_document(i)
        if doc is None:
            continue
        if noskip:
            if len(doc) >= min_doc_character_len:
                doc = doc.strip() + '\n'
                yield doc
        else:
            for x in skip_all_sentences_with_banned_words(doc):
                if len(x) >= min_doc_character_len:
                    yield x


encoder = get_encoder()


def buffer_shuffler_iterator(inds, seq_len, min_doc_character_len=64, buffer_size=10000,
                             noskip=False):
    """
    What we want to do is start with a sequence, and then sample a continuation short sequence at random
    :param inds: Indices for going through all documents
    :param seq_len: The length of each sequence
    :param min_doc_character_len: If a document is lower than this #, skip it.
    :param buffer_size: Size of the buffer for shuffling
    :param noskip:
    :return: Documents of exactly [seq_len]
    """
    buffer = []

    def _pop_from_buffer():
        """
        If the first item is longer, yield from a random sliding window, otherwise if the first item is shorter,
        combine it with later items. Pop everything needed.

        WHAT THE FUCK
        https://stackoverflow.com/questions/5218895/python-nested-functions-variable-scoping
        """
        nonlocal buffer

        random.shuffle(buffer)
        assert len(buffer) > 0
        t0 = buffer.pop(0)

        if len(t0) >= seq_len:
            yield from random_sliding_window(t0, max_seq_length=seq_len)
        elif len(buffer) == 0:
            # We already checked that t0 has length at most seq_len
            yield t0
        else:
            for i, b_i in enumerate(buffer):
                if len(t0) >= seq_len:
                    break
                else:
                    t0 += [encoder.reset_context] + b_i
            # Now pop all of the articles that we "fully used"
            buffer = buffer[max(i - 1, 0):]
            yield t0[:seq_len]

    for doc in document_iterator(inds, min_doc_character_len=min_doc_character_len, noskip=noskip):
        # Append (a) the encoded document and (b) if it's the real end of the document
        # which I encoded by '\n'
        is_real_end = doc.endswith('\n')
        doc_enc = encoder.encode(doc.strip())

        if is_real_end:
            doc_enc = [encoder.begin_article] + doc_enc + [encoder.end_article]
        else:
            doc_enc = [encoder.begin_article] + doc_enc + [encoder.padding]
        buffer.append(doc_enc)

        if len(buffer) > buffer_size:
            while len(buffer) > buffer_size * 0.9:
                yield from _pop_from_buffer()

    while len(buffer) > 0:
        yield from _pop_from_buffer()
