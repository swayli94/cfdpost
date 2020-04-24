
from cfdpost.feature2d import FeatureSec
from cfdpost.cfdresult import cfl3d


def post_foil_cfl3d(path, j0, j1, nHi=40, fname="feature2d.txt"):
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


