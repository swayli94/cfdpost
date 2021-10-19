
import time

from cfdpost.cfdresult import cfl3d
from cfdpost.section.physical import PhysicalSec


def post_foil_cfl3d(path, j0, j1, nHi=40, fname='feature2d.txt', tecplot=False):
    '''
    Read in CFL3D foil result and extract flow features.

    Inputs:
    ---
    path:   folder that contains the output files \n
    j0:     j index of the lower surface TE \n
    j1:     j index of the upper surface TE \n
    nHi:    maximum number of mesh points in k direction for boundary layer \n
    fname:  output file name. If None, then no output \n
    tecplot:    True, then convert cfl3d.prt to surface.dat \n

    Single C-block cfl3d.prt index
    ---
    ```text
    i : 1 - 1   symmetry plane
    j : 1 - nj  far field of lower surface TE to far field of upper surface TE
    k : 1 - nk  surface to far field
    ```
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

    Hi, Hc, info = PhysicalSec.getHi(X,Y,U,V,T,j0=j0,j1=j1,nHi=nHi)
    (Tw, dudy) = info

    fF = PhysicalSec(Minf, AoA, Re*1e6)
    fF.setdata(x, y, Cp1, Tw, Hi, Hc, dudy)
    fF.extract_features()

    if fname is not None:
        fF.output_features(fname=fname, append=True, keys_=None)

    if tecplot:
        succeed = cfl3d.readprt('path')

    return fF


if __name__ == "__main__":

    t0 = time.perf_counter()
    path = 'cfl3d-foil'

    iVar = [1,2,3,5,6,7,9,11]
    var_name = ['density', 'u', 'v', 'e', 'velocity', 'T', 'M', 'cp']

    xy, qq, mach, alfa, reyn = cfl3d.readPlot2d(path)
    
    variables = cfl3d.analysePlot3d(mach, qq, iVar, gamma_r=1.4)

    cfl3d.outputTecplot(xy, variables, var_name, fname='flow-field.dat', append=False)

    cfl3d.readprt(path='cfl3d-foil')

    #post_foil_cfl3d(path, j0=40, j1=341, fname="feature2d.txt")

    t1 = time.perf_counter()
    print('Time = %.3f s'%(t1-t0))
