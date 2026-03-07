from py_electrodes.py_electrodes import PyElectrode

if __name__ == '__main__':
    pe = PyElectrode("Brep Electrode")
    pe.generate_from_file("entrance_plate.stl")
    pe.show()
