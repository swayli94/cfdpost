
import time

import numpy as np
from cfdpost.section.physical import PhysicalXfoil


def feature_xfoil(cst_u, cst_l, t, Minf: float, Re, AoA, n_crit=0.1, fname='feature-xfoil.txt'):
    '''
    Evaluate by xfoil and extract features.

    Inputs:
    ---
    cst-u, cst-l:   list of upper/lower CST coefficients of the airfoil. \n
    t:      airfoil thickness or None \n
    Minf:   free stream Mach number for wall Mach number calculation \n
    Re, AoA (deg): flight condition (s), float or list, for Xfoil \n
    n_crit: critical amplification ratio for transition in xfoil \n
    fname:  output file name. If None, then no output \n

    ### Dependencies: cst-modeling3d, xfoil  
    '''

    from cst_modeling.foil import cst_foil
    from xfoil import XFoil
    from xfoil.model import Airfoil

    #TODO: Build foil
    #! 201 is the maximum amount of points that xfoil can handle
    #! tail = 0.001 is to avoid point overlap
    xx, yu, yl, t0, R0 = cst_foil(201, cst_u, cst_l, x=None, t=t, tail=0.001)

    #! xfoil do not support leading edge of (0,0) on both upper and lower surface
    x = np.array(list(reversed(xx[1:])) + xx[1:])
    y = np.array(list(reversed(yu[1:])) + yl[1:])
    foil = Airfoil(x, y)

    #TODO: Xfoil
    xf = XFoil()
    xf.print = False
    xf.airfoil = foil
    xf.max_iter = 40

    #* Transition by power law
    xf.n_crit = n_crit

    #TODO: Xfoil calculation
    if not isinstance(Re, list):
        Re = [Re]
        AoA = [AoA]

    n = len(Re)
    for i in range(n):

        xf.reset_bls()

        if Re[i] is not None:
            xf.Re = Re[i]
        
        cl, cd, cm, cp = xf.a(AoA[i])
        x, cp = xf.get_cp_distribution()

        print(xf.Re, AoA[i], cl)

        #* Extract features
        fF = PhysicalXfoil(Minf, AoA[i], Re[i])
        fF.setdata(x,y,cp)
        fF.extract_features()

        #* Output
        if fname is None:
            continue

        if i == 0:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')
        
        f.write('\n')
        f.write('%10s   %15.6f \n'%('Minf', Minf))
        f.write('%10s   %15.6f \n'%('AoA', AoA[i]))
        f.write('%10s   %15.6f \n'%('Re', Re[i]/1e6))
        f.write('%10s   %15.6f \n'%('CL', cl))
        f.write('%10s   %15.6f \n'%('Cd', cd))
        f.write('%10s   %15.6f \n'%('Cm', cm))
        f.close()

        fF.output_features(fname=fname, append=True)


if __name__ == "__main__":

    t0 = time.perf_counter()

    #TODO: CST airfoil for Xfoil
    cst_u = [ 0.135283,  0.088574,  0.177210,  0.080000,  0.231590,  0.189572,  0.192000]
    cst_l = [-0.101390, -0.007993, -0.240000, -0.129790, -0.147840, -0.000050,  0.221251]
    t = 0.0954
    Minf = 0.2
    AoA  = [5.0, 6.0]
    Re   = [5e5, 5e5]
    feature_xfoil(np.array(cst_u), np.array(cst_l), t, Minf, Re, AoA, n_crit=0.1, fname='feature-xfoil.txt')

    t1 = time.perf_counter()
    print('Time = %.3f s'%(t1-t0))
