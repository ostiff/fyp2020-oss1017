# TODO: Add method to zip log on destructor call

# Libraries
import os
import time
from pathlib import Path
import pickle

from definitions import LOG_PATH

class Logger:

    def __init__(self, log_type, compress=False):
        self._log_type = log_type
        self._compress = compress

        # Create log directory
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self._path = Path(os.path.join(LOG_PATH, log_type, timestr))
        self._path.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        if os.path.exists(self._path) and os.path.isdir(self._path):

            # If log dir is empty remove it
            if not os.listdir(self._path):
                self._path.rmdir()
            else:
                if self.compress:
                    pass

    @property
    def compress(self):
        return self._compress

    @property
    def log_type(self):
        return self._log_type

    @property
    def path(self):
        return self._path

    def save_model(self, model, filename='model'):
        file_path = os.path.join(self._path, filename)
        pickle.dump(model, open(file_path, 'wb'))

    def create_report(self):
        pass
