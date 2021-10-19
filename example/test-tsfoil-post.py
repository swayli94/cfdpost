
import time

import numpy as np
from cfdpost.section.physical import PhysicalTSFoil


def feature_TSFoil(cst_u, cst_l, t, Minf, Re, AoA, fname='feature-xfoil.txt'):
    '''
    Evaluate by TSFOIL2 and extract features.

    Inputs:
    ---
    cst-u, cst-l:   list of upper/lower CST coefficients of the airfoil \n
    t:      airfoil thickness or None \n
    Minf, Re, AoA (deg): flight condition (s), float or list \n
    fname:  output file name. If None, then no output \n

    ### Dependencies: cst-modeling3d, pyTSFoil
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
        fF = PhysicalTSFoil(Minf[i], AoA[i], Re[i])
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


if __name__ == "__main__":

    t0 = time.perf_counter()

    #TODO: CST airfoil for Xfoil
    cst_u = [ 0.135283,  0.088574,  0.177210,  0.080000,  0.231590,  0.189572,  0.192000]
    cst_l = [-0.101390, -0.007993, -0.240000, -0.129790, -0.147840, -0.000050,  0.221251]
    t = 0.0954
    Minf = 0.76
    AoA  = 0.6
    Re   = 5e5
    feature_TSFoil(np.array(cst_u), np.array(cst_l), t, Minf, Re, AoA, fname='feature-tsfoil.txt')

    t1 = time.perf_counter()
    print('Time = %.3f s'%(t1-t0))
