"""
Logger class implementation.
"""

# Libraries
import os
import re
import csv
import time
import atexit
import zipfile
from functools import wraps
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import shutil

from definitions import LOG_PATH, TEMPLATES_PATH


class Logger:
    """Class used to log parameters and save results to an archive
    or directory.
    """

    def __init__(self, log_type, compress=True, enable=True):
        """Constructor method.

        :param str log_type: String used in path to identify the type of log.
        :param bool compress: Compress the log.
        :param bool enable: Enable logging.
        """
        self.enable = enable \
                      and os.path.exists(LOG_PATH) \
                      and os.path.isdir(LOG_PATH)

        if self.enable:
            self._log_type = log_type
            self._compress = compress
            self._parameters = dict()
            self._report = list()
            self._figcount = 0
            # Create log directory
            self._time = time.localtime()
            self._path = Path(os.path.join(LOG_PATH, log_type, time.strftime("%Y%m%d-%H%M%S", self._time)))
            self._path.mkdir(parents=True, exist_ok=True)

            # Call on_exit when obj is garbage collected
            atexit.register(self.on_exit)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        atexit.unregister(self.on_exit())
        self.on_exit()

    def _log_enable(func):
        """Decorator to only execute class methods if logging is enabled.
        """

        @wraps(func)
        def wrapped(inst, *args, **kwargs):
            if inst.enable:
                return func(inst, *args, **kwargs)

        return wrapped

    @_log_enable
    def on_exit(self):
        """Called when class object is garbage collected. Compresses
        the log if compression is enabled.
        """
        if os.path.exists(self._path) and os.path.isdir(self._path):

            # If log dir is empty remove it
            if not os.listdir(self._path):
                self._path.rmdir()
            else:
                if self._compress:
                    shutil.make_archive(self._path, 'zip', self._path)
                    shutil.rmtree(self._path)

    @_log_enable
    @property
    def compress(self):
        """Returns value of `compress`.

        :return bool:
        """
        return self._compress

    @_log_enable
    @property
    def log_type(self):
        """Returns log type.

        :return str:
        """
        return self._log_type

    @_log_enable
    @property
    def path(self):
        """Return path to log.

        :return pathlib.Path:
        """
        return self._path

    @_log_enable
    def save_object(self, obj, filename='model'):
        """Serialises object and saves it to the log.

        :param obj: Object to serialise.
        :param filename:
        """
        file_path = os.path.join(self._path, filename)
        pickle.dump(obj, open(file_path, 'wb'))

    @_log_enable
    def add_parameters(self, parameters):
        """Add parameters to the Logger object.
        `save_parameters` must be called to add these to the report.

        :param parameters: Dictionary of parameter names -> values.
        """
        self._parameters = {**self._parameters,
                            **parameters}

    @_log_enable
    def save_parameters(self, parameters=None):
        """Add parameters to Logger object and add them to the body of the
        generated report.
        Save parameters to a JSON file.

        :param dict() parameters: Dictionary of parameter names -> values.
        """
        if parameters is not None:
            self.add_parameters(parameters)

        file_path = os.path.join(self._path, 'parameters.json')

        with open(file_path, 'w') as fp:
            json.dump(self._parameters, fp, indent=4)

    @_log_enable
    def add_plt(self, plot, fname='figure', ext='svg'):
        """Add a matplotlib figure to the body of the generated report.

        :param matplotlib.figure.Figure plot: matplotlib figure.
        :param str fname: Filename (not full path).
        :param str ext: File extension. (Avoid svg for big figures)
        """
        fname += f'{self._figcount}.{ext}'
        self._figcount += 1
        plot.savefig(os.path.join(self._path, fname))
        html = f'<img src="./{fname}">'
        self.append_html(html)

    @_log_enable
    def append_html(self, html):
        """Add an HTML element to the body of the generated report.

        :param str html: HTML input in string form
        """
        self._report.append(html)

    @_log_enable
    def create_report(self):
        """Creates an HTML report combining parameters and all plots/HTML
        elements added to the Logger object.
        """
        with open(os.path.join(TEMPLATES_PATH, 'log_report_style.html'), 'r') as f:
            style = f.read()
        html = f"""
                <html>
                <head>
                <title>{self._log_type}</title>
                {style}
                </head>
                <body>
                <h1>{self._log_type} - {time.strftime('%b %d %Y, %H:%M:%S', self._time)}</h1>
                <br>\n
                """

        if self._parameters:
            html += "<h2>Parameters:</h2>"
            html += "<table id=><tr><th>Parameter</th><th>Value</th></tr>\n"

            for k, v in self._parameters.items():
                html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"

            html += "</table><br>\n"

        if self._report:
            html += "<hr><br><h2>Output:</h2>"
            for fig in self._report:
                html += fig
                html += "<br><hr>"

        html += """
                </body>
                </html>
                """

        file_path = os.path.join(self._path, 'report.html')
        with open(file_path, "w") as file:
            file.write(html)


def extract_log_results(dir_path, keys, fpath='compiled_logs.csv', verbose=False):
    count = 0
    keys.insert(0, 'log_name')

    with open(fpath, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, extrasaction="ignore")
        dict_writer.writeheader()

        for subdir, dirs, files in os.walk(dir_path):
            for filename in files:
                filepath = subdir + os.sep + filename

                if re.match(r"^[0-9]{8}-[0-9]{6}(.zip)$", filename):
                    archive = zipfile.ZipFile(filepath)
                    archive_fileList = archive.namelist()

                    for archive_filename in archive_fileList:
                        if archive_filename.endswith('.json'):
                            try:
                                json_data = json.loads(archive.read(archive_filename))
                                json_data['log_name'] = filename.rsplit('.', 1)[0]
                                dict_writer.writerow(json_data)
                                if verbose:
                                    print(f"Added {filepath} - {archive_filename}")
                                count += 1
                            except Exception as e:
                                print(f"Could not add {filepath} - {archive_filename}\n{e}")

                    archive.close()

                elif re.match(r"^.*[0-9]{8}-[0-9]{6}$", subdir) and filename.endswith('.json'):
                    try:
                        with open(filepath, 'r') as json_file:
                            json_data = json.loads(json_file.read())
                            json_data['log_name'] = subdir.rsplit(os.sep)[-1]
                        dict_writer.writerow(json_data)
                        if verbose:
                            print(f"Added {filepath}")
                        count += 1
                    except Exception as e:
                        print(f"Could not add {filepath}\n{e}")

    if verbose:
        print(f"{count} rows added to {fpath}.")


def get_log_keys(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        for filename in files:
            filepath = subdir + os.sep + filename

            if re.match(r"^[0-9]{8}-[0-9]{6}(.zip)$", filename):
                archive = zipfile.ZipFile(filepath)
                archive_fileList = archive.namelist()

                for archive_filename in archive_fileList:
                    if archive_filename.endswith('.json'):
                        try:
                            json_data = json.loads(archive.read(archive_filename))
                            return [*json_data]
                        except Exception as e:
                            print(f"Could not read {filepath} - {archive_filename}\n{e}")

                archive.close()

            elif re.match(r"^.*[0-9]{8}-[0-9]{6}$", subdir) and filename.endswith('.json'):
                with open(filepath) as json_file:
                    try:
                        json_data = json.loads(json_file)
                        return [*json_data]
                    except Exception as e:
                        print(f"Could not read {filepath}\n{e}")

    return None


if __name__ == '__main__':
    keys = get_log_keys(r'C:\Users\Oliver\Desktop\desktop\RESULTS\TSNE_Dengue_grid_search')
    extract_log_results(r'C:\Users\Oliver\Desktop\desktop\RESULTS\TSNE_Dengue_grid_search',
                        keys, fpath='ae_grid_search.csv', verbose=True)
