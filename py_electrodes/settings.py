import os
import sys
import platform
import pickle


class SettingsHandler(object):
    """
    Class that handles the Settings.txt file. In both Windows and Linux, there are two locations:
    Preferred: $HOME/.local/lib/python<version>/site-packages/py_electrodes (Linux)
               $APPDATA/py_electrodes (Windows)  (usually C:\Users\<username>\AppData\Roaming)
    Fallback: Wherever the module is installed
              e.g. ...Anaconda3\envs\<env name>\Lib\site-packages\py_electrodes
    The handler looks in the preferred directory first, so the user can override the default settings
    by creating the py_electrodes directory and putting a copy of Settings.txt in there.
    """

    def __init__(self):

        # Are we on Windows or Linux?
        self._unix = True
        if "win" in platform.platform().lower():
            self._unix = False

        if self._unix:
            self._path1 = "{}/.local/lib/python{}.{}/site-packages/py_electrodes".format(os.environ["HOME"],
                                                                                         sys.version_info.major,
                                                                                         sys.version_info.minor)
        else:
            self._path1 = os.path.join("{}".format(os.environ['APPDATA']), "py_electrodes")

        self._path2 = os.path.abspath(os.path.dirname(__file__))

        self._settings = {"DEBUG": False,
                          "DECIMALS": 12,
                          "TEMP_DIR": os.path.join(self._path2, "temp"),
                          "GMSH_EXE": "gmsh"}

        # Immediately load settings upon creation
        self.load_settings()

    def __getitem__(self, item):
        # return item by key or None if it doesn't exist
        return self._settings.get(item, None)

    def load_settings(self):
        """
        This function loads the settings from one of two locations:
        Preferred: $HOME/.local/lib/python<version>/site-packages/py_electrodes
        Fallback: <package-dir>
        If neither exist, it will write a new Settings.txt file into the package dir.
        :return:
        """

        filepath1 = os.path.join(self._path1, "Settings.txt")
        filepath2 = os.path.join(self._path2, "Settings.txt")

        print(filepath1)
        print(filepath2)

        if os.path.isfile(filepath1):
            print("Filepath1 exists")
            self.read_from_file(filepath1)
        elif os.path.isfile(os.path.join(self._path2, "Settings.txt")):
            print("Filepath2 exists")
            self.read_from_file(filepath2)
        else:
            assert os.path.isdir(self._path2), "Could not find default fallback directory to put Settings.txt file in!"
            print("No Settings.txt file found, creating in {}".format(filepath2))
            self.create_settings_file(filepath2)

    def read_from_file(self, filepath):
        with open(filepath, "r") as _if:
            data = pickle.load(_if)
        for _key in self._settings.keys():
            self._settings[_key] = data.get(_key, self._settings[_key])

    def create_settings_file(self, filepath):
        with open(filepath, "w") as _of:
            pickle.dump(self._settings, _of)
