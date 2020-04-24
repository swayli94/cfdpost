import copy
import time

from cfdpost.post import post_foil_cfl3d

if __name__ == "__main__":

    t0 = time.perf_counter()
    path = '.\example\cfl3d-foil'

    post_foil_cfl3d(path, j0=40, j1=341, fname="feature2d.txt")

    t1 = time.perf_counter()
    print('Time = %.3f s'%(t1-t0))
