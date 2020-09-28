import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from cst_modeling.foil import cst_foil
from xfoil import XFoil
from xfoil.model import Airfoil

from cfdpost.section.physical import PhysicalXfoil

if __name__ == "__main__":

    t0 = time.perf_counter()

    #TODO: CST airfoil for Xfoil
    cst_u = [ 0.135283,  0.088574,  0.177210,  0.080000,  0.231590,  0.189572,  0.192000]
    cst_l = [-0.101390, -0.007993, -0.240000, -0.129790, -0.147840, -0.000050,  0.221251]
    xx, yu, yl, t0, R0 = cst_foil(101, np.array(cst_u), np.array(cst_l), x=None, t=0.0954, tail=0.002)

    x = np.concatenate((np.flip(xx[1:]), xx[1:]), axis=0)
    y = np.concatenate((np.flip(yu[1:]), yl[1:]), axis=0)
    foil = Airfoil(x, y)

    #TODO: Xfoil
    xf = XFoil()
    xf.print = False
    xf.max_iter = 40
    xf.airfoil = foil
    xf.xtr = [0.0, 0.0]

    Minf = 0.2
    AoA  = 8.0
    Re   = 1e7
    fname = 'feature-xfoil.txt'

    xf.M = Minf
    xf.Re = Re

    cl, cd, cm, cp = xf.a(AoA)
    x, cp = xf.get_cp_distribution()

    with open(fname, 'w') as f:
        f.write('%10s   %15.6f \n'%('Minf', Minf))
        f.write('%10s   %15.6f \n'%('AoA', AoA))
        f.write('%10s   %15.6f \n'%('Re', Re/1e6))
        f.write('%10s   %15.6f \n'%('CL', cl))
        f.write('%10s   %15.6f \n'%('Cd', cd))
        f.write('%10s   %15.6f \n'%('Cm', cm))

    fF = PhysicalXfoil(Minf, AoA, Re)
    fF.setdata(x,y,cp)
    fF.extract_features()
    fF.output_features(fname=fname, append=True)

    t1 = time.perf_counter()
    print('Time = %.3f s'%(t1-t0))

#   exit()

    plt.figure()
    plt.plot(fF.x, fF.Mw)
    plt.show()