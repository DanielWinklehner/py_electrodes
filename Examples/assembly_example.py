from py_electrodes.py_electrodes import PyElectrode, PyElectrodeAssembly
import os

if __name__ == '__main__':

    filenames = ["entrance_plate.geo",
                 "exit_plate.geo",
                 "vane_xm.geo",
                 "vane_ym.geo",
                 "vane_xp.geo",
                 "vane_yp.geo"]

    pa = PyElectrodeAssembly("RFQ with endplates")

    for _fn in filenames:

        pe = PyElectrode(os.path.splitext(_fn)[0])

        if "_y" in _fn:
            pe.color = 'RED'
        elif "_x" in _fn:
            pe.color = 'BLUE'
        else:
            pe.color = 'GREEN'

        pe.generate_from_file(_fn)
        pa.add_electrode(pe)

    pa.show()

    pa.get_bempp_mesh()
