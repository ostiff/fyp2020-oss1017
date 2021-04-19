# TODO: Add method to zip log on destructor call

# Libraries
import os
import time
import atexit
from pathlib import Path
import pickle
import json

from definitions import LOG_PATH, TEMPLATES_PATH

class Logger:
    _parameters = dict()
    _report = []

    def __init__(self, log_type, compress=False):
        self._log_type = log_type
        self._compress = compress

        # Create log directory
        self._time = time.localtime()
        self._path = Path(os.path.join(LOG_PATH, log_type, time.strftime("%Y%m%d-%H%M%S", self._time)))
        self._path.mkdir(parents=True, exist_ok=True)

        # Call on_exit when obj is garbage collected
        atexit.register(self.on_exit)

    def on_exit(self):
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

    def save_object(self, model, filename='model'):
        file_path = os.path.join(self._path, filename)
        pickle.dump(model, open(file_path, 'wb'))

    def add_parameters(self, parameters):
        self._parameters = {**self._parameters,
                            **parameters}

    def save_parameters(self, parameters=None):
        if parameters is not None:
            self.add_parameters(parameters)

        file_path = os.path.join(self._path, 'parameters.json')

        with open(file_path, 'w') as fp:
            json.dump(self._parameters, fp, indent=4)

    def append_html(self, html):
        self._report.append(html)

    def create_report(self):
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
