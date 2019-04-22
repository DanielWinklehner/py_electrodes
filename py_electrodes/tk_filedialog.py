import sys
import os
if sys.version_info.major == 3:
    from tkinter import filedialog
    from tkinter import *
    py_ver = 3
elif sys.version_info.major == 2:
    from Tkinter import *
    import tkFileDialog
    py_ver = 2
else:
    raise Exception("This version of python is not recognized!")

__author__ = "Daniel Winklehner"
__doc__ = """A wrapper around tkinter filedialogs"""


class FileDialog(object):

    def __init__(self):

        self._filename = None
        self._parent = None
        self._icon = None
        self._root = None

    def get_filename(self, action='open', old_path=None, icon=None, parent=None):

        self._parent = parent
        self._icon = icon
        self._root = Tk()
        self._root.withdraw()

        if py_ver == 2:
            myfiledialog = tkFileDialog
        else:
            myfiledialog = filedialog

        if action == 'open':

            self._filename = myfiledialog.askopenfilename(initialdir=old_path,
                                                          title='Open file...',
                                                          parent=self._root)

        elif action == 'save':

            self._filename = myfiledialog.asksaveasfilename(initialdir=old_path,
                                                            title='Save as...')

        elif action == 'folder':
            self._filename = myfiledialog.askdirectory(initialdir=old_path,
                                                       title='Select folder (manually write new one in path string)')

            if self._filename is not None and not os.path.exists(self._filename) and self._filename != '':
                os.makedirs(self._filename)

        else:

            raise Exception("'action' must be 'open', 'save', or 'folder' (got '%s')" % action)

        self._root.destroy()

        if not self._filename:
            self._filename = None

        return self._filename
