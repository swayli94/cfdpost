
import time

from cfdpost.feature2d import FeatureXfoil
from cfdpost.post import feature_xfoil


if __name__ == "__main__":

    t0 = time.perf_counter()

    #TODO: CST airfoil for Xfoil
    cst_u = [ 0.135283,  0.088574,  0.177210,  0.080000,  0.231590,  0.189572,  0.192000]
    cst_l = [-0.101390, -0.007993, -0.240000, -0.129790, -0.147840, -0.000050,  0.221251]
    t = 0.0954
    Minf = 0.2
    AoA  = [5.0, 6.0]
    Re   = [5e5, 5e5]
    feature_xfoil(cst_u, cst_l, t, Minf, Re, AoA, n_crit=0.1, fname='feature-xfoil.txt')

    t1 = time.perf_counter()
    print('Time = %.3f s'%(t1-t0))
