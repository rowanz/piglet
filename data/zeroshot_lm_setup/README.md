# zero shot LM setup

---------------

Here's the procedure
* Get the word list using `get_word_list.py`. Requires a lot of manual involvement
* Create language only tfrecords from google books / wikipedia
    * for Wikipedia 
```
import nlp
wikipedia = nlp.load_dataset('wikipedia', '20200501.en')['train']
```
Run this command ahead of time.
There are 5354968 over 2048 folds so 2614 per.


### Running this
Something like...
```bash
parallel -j 12 --will-cite "python prepare_zslm_tfrecord.py -fold={1} -num_folds=${NUM_FOLDS} -out_path gs://OUTPATH/ > logs/train{1}.txt" ::: $(seq ${STARTFOLD} ${ENDFOLD})
```