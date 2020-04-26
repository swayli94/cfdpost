
import numpy as np

from cfdpost.cfdresult import cfl3d
from cfdpost.feature2d import FeatureSec, FeatureXfoil, FeatureTSFoil

import matplotlib.pyplot as plt


def post_foil_cfl3d(path, j0, j1, nHi=40, fname='feature2d.txt', tecplot=False):
    '''
    Read in CFL3D foil result and extract flow features. \n
        path:   folder that contains the output files
        j0:     j index of the lower surface TE
        j1:     j index of the upper surface TE
        nHi:    maximum number of mesh points in k direction for boundary layer
        fname:  output file name. If None, then no output
        tecplot:    True, then convert cfl3d.prt to surface.dat

    Single C-block cfl3d.prt index \n
        i : 1 - 1   symmetry plane
        j : 1 - nj  far field of lower surface TE to far field of upper surface TE
        k : 1 - nk  surface to far field
    '''

    converge, CL, CD, Cm, CDp, CDf = cfl3d.readCoef(path)

    if not converge:
        print('  CFD not converged. '+path)
        return

    succeed1, AoA = cfl3d.readAoA(path)
    succeed2, Minf, AoA0, Re, l2D = cfl3d.readinput(path)
    succeed3, field, foil = cfl3d.readprt_foil(path, j0=j0, j1=j1)

    if not succeed1:
        AoA = AoA0

    print(Minf, AoA, Re*1e6, CL)

    if not succeed2 or not succeed3:
        print('  CFD output file corrupted. '+path)
        return
    
    (x, y, Cp1) = foil
    (X,Y,U,V,P,T,Ma,Cp,vi) = field

    Hi, Hc, info = FeatureSec.getHi(X,Y,U,V,T,j0=j0,j1=j1,nHi=nHi)
    (Tw, dudy) = info

    fF = FeatureSec(Minf, AoA, Re*1e6)
    fF.setdata(x, y, Cp1, Tw, Hi, Hc, dudy)
    fF.extract_features()

    if fname is not None:
        fF.output_features(fname=fname, append=True, keys_=None)

    if tecplot:
        succeed = cfl3d.readprt('path')

    return fF

def feature_xfoil(cst_u: list, cst_l: list, t, Minf: float, Re, AoA, n_crit=0.1, fname='feature-xfoil.txt'):
    '''
    Evaluate by xfoil and extract features. \n
        cst-u, cst-l:   list of upper/lower CST coefficients of the airfoil.
        t:      airfoil thickness or None
        Minf:   free stream Mach number for wall Mach number calculation
        Re, AoA (deg): flight condition (s), float or list, for Xfoil
        n_crit: critical amplification ratio for transition in xfoil
        fname:  output file name. If None, then no output

    Dependencies: \n
        cst-modeling3d, xfoil
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
        fF = FeatureXfoil(Minf, AoA[i], Re[i])
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

    return fF

def feature_TSFoil(cst_u: list, cst_l: list, t, Minf, Re, AoA, fname='feature-xfoil.txt'):
    '''
    Evaluate by TSFOIL2 and extract features. \n
        cst-u, cst-l:   list of upper/lower CST coefficients of the airfoil
        t:      airfoil thickness or None
        Minf, Re, AoA (deg): flight condition (s), float or list
        fname:  output file name. If None, then no output

    Dependencies: \n
        cst-modeling3d, pyTSFoil
    '''

    from pyTSFoil.TSfoil import TSfoil

    ts = TSfoil()
    ts.foil_byCST(cst_u, cst_l, t=t)

    #TODO: Calculation
    if not isinstance(Minf, list):
        Minf = [Minf]
        Re = [Re]
        AoA = [AoA]

    n = len(Re)
    for i in range(n):

        ts.flight_condition(Minf[i], AoA[i], Re=Re[i], wc=4.0)
        ts.run(show=False)
        ts.get_result()

        print(Minf[i], AoA[i], Re[i], ts.CL)

        #* Extract features
        fF = FeatureTSFoil(Minf[i], AoA[i], Re[i])
        fF.setdata(ts.xu, ts.yu, ts.xl, ts.yl, ts.cpu, ts.cpl, ts.mwu, ts.mwl)
        fF.extract_features()

        #* Output
        if fname is None:
            continue

        if i == 0:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')
        
        f.write('\n')
        f.write('%10s   %15.6f \n'%('Minf', Minf[i]))
        f.write('%10s   %15.6f \n'%('AoA', AoA[i]))
        f.write('%10s   %15.6f \n'%('Re', Re[i]/1e6))
        f.write('%10s   %15.6f \n'%('CL', ts.CL))
        f.write('%10s   %15.6f \n'%('Cd', ts.Cd))
        f.write('%10s   %15.6f \n'%('Cdw', ts.Cdw))
        f.write('%10s   %15.6f \n'%('Cm', ts.Cm))
        f.close()

        fF.output_features(fname=fname, append=True)

    return fF










