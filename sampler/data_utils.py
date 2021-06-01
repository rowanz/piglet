from google.cloud import storage
import tempfile
import h5py
import os

class GCSH5Writer(object):
    def __init__(self, fn):
        self.fn = fn
        if fn.startswith('gs://'):
            self.gclient = storage.Client()
            self.storage_dir = tempfile.TemporaryDirectory()
            self.writer = h5py.File(os.path.join(self.storage_dir.name, 'temp.h5'), 'w')
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            assert not os.path.exists(self.fn)
            self.writer = h5py.File(self.fn, 'w')

    def create_group(self, name, track_order=None):
        return self.writer.create_group(name, track_order=track_order)

    def create_dataset(self, name, data, **kwargs):
        return self.writer.create_dataset(name, data=data, **kwargs)

    def close(self):
        self.writer.close()

        if self.gclient is not None:
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.h5'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        print("CALLING CLOSE")
        self.close()