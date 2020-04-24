
import numpy as np

from cfdpost.cfdresult import cfl3d
from cfdpost.feature2d import FeatureSec, FeatureXfoil


def post_foil_cfl3d(path, j0, j1, nHi=40, fname='feature2d.txt'):
    '''
    Read in CFL3D foil result and extract flow features. \n
        path:   folder that contains the output files
        j0:     j index of the lower surface TE
        j1:     j index of the upper surface TE
        nHi:    maximum number of mesh points in k direction for boundary layer

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

    if not succeed2 or not succeed3:
        print('  CFD output file corrupted. '+path)
        return
    
    (x, y, Cp1) = foil
    (X,Y,U,V,P,T,Ma,Cp,vi) = field

    Hi, Hc, info = FeatureSec.getHi(X,Y,U,V,T,j0=j0,j1=j1,nHi=nHi)
    (Tw, dudy) = info

    fF = FeatureSec(Minf, AoA, Re)
    fF.setdata(x, y, Cp1, Tw, Hi, Hc, dudy)
    fF.extract_features()
    fF.output_features(fname=fname, append=True, keys_=None)

    # succeed = cfl3d.readprt('path')

def feature_xfoil(cst_u: list, cst_l: list, t: float, Minf: float, Re, AoA, n_crit=0.1, fname='feature-xfoil.txt'):
    '''
    Evaluate by xfoil and extract features. \n
        cst-u, cst-l:   list of upper/lower CST coefficients of the airfoil.
        t:      airfoil thickness or None
        Minf:   free stream Mach number for wall Mach number calculation
        Re, AoA (deg): flight condition (s), float or list, for Xfoil
        n_crit: critical amplification ratio for transition in xfoil
        fname:  output file.

    Dependencies: \n
        cst_modeling, xfoil
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










