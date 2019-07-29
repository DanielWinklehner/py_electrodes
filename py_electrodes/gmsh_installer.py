import os
import sys
import zipfile
import tarfile


class GmshInstaller(object):

    def __init__(self):
        self._windows_machine = "win" in sys.platform.lower()
        self._gmsh_zip_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "resources")

        if self._windows_machine:
            self._gmsh_zip = os.path.join(self._gmsh_zip_path, "gmsh-4.4.1-Windows64.zip")
            self._gmsh_exe_path = os.path.join(self._gmsh_zip_path, "gmsh-4.4.1-Windows64")
            self._gmsh_exe = os.path.join(self._gmsh_exe_path, "gmsh.exe")
        else:
            self._gmsh_zip = os.path.join(self._gmsh_zip_path, "gmsh-4.4.1-Linux64.tgz")
            self._gmsh_exe_path = os.path.join(self._gmsh_zip_path, "gmsh-4.4.1-Linux64", "bin")
            self._gmsh_exe = os.path.join(self._gmsh_exe_path, "gmsh")

    def run(self):
        if os.path.isfile(self._gmsh_exe):
            print("GmshInstaller found gmsh 4.4.1 in py_electrodes module. Using it!")
            return self._gmsh_exe
        elif os.path.isfile(self._gmsh_zip):
            print("GmshInstaller could not find gmsh exe, but py_electrodes shipped with zipped gmsh. Unpacking...")

            self.extract_zip()

            if os.path.isfile(self._gmsh_exe):
                print("GmshInstaller found gmsh 4.4.1 in py_electrodes module. Using it!")
                return self._gmsh_exe

            else:
                print("Something went wrong during gmsh installation! Please install manually and set the "
                      "correct path in Settings.txt")
                return None

        else:
            print("Couldn't find gmsh zip file in py_electrodes installation! Please install gmsh manually and set the "
                  "correct path in Settings.txt")
            return None

    def extract_zip(self):
        if self._windows_machine:
            with zipfile.ZipFile(self._gmsh_zip, "r") as zip_ref:
                zip_ref.extractall(self._gmsh_zip_path)
        else:
            with tarfile.open(self._gmsh_zip, "r:") as _tarfile:
                _tarfile.extractall(self._gmsh_zip_path)


if __name__ == "__main__":
    pass
