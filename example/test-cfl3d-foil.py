import copy
import time

from cfdpost.post import post_foil_cfl3d
from cfdpost.cfdresult import cfl3d

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
