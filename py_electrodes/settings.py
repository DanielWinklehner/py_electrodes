import os
import sys


class SettingsHandler(object):
    def __init__(self):
        self._path1 = "$HOME/.local/lib/python{}.{}/site-packages/py_electrodes".format(sys.version_info.major,
                                                                                        sys.version_info.minor)
        self._path2 = os.path.abspath(os.path.dirname(__file__))

    def load_from_file(self):
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
        elif os.path.isfile(os.path.join(self._path2, "Settings.txt")):
            print("Filepath2 exists")
        else:
            assert os.path.isdir(self._path2), "Could not find default fallback directory to put Settings.txt file in!"
            print("No Settings.txt file found, creating in {}".format(filepath2))
