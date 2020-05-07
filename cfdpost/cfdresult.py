'''
Post process of CFD results
'''

import os
import platform
import numpy as np

class cfl3d():
    '''
    Extracting data from cfl3d results
    '''

    def __init__(self):
        print('All static method functions')
        pass

    @staticmethod
    def readCoef(path: str, n=10):
        '''
        Read clcd_wall.dat or clcd.dat of the CFL3D outputs. \n
            path:   folder that contains the results
            n:      get the mean value of final n steps

        Return: \n
            converge (bool), CL, CD, Cm(z), CDp, CDf
        '''
        converge = True
        CL = 0.0
        CD = 0.0
        Cm = 0.0
        CDp = 0.0
        CDf = 0.0

        if platform.system() in 'Windows':
            out1 = path+'\\clcd.dat'
            out2 = path+'\\clcd_wall.dat'

        else:
            out1 = path+'/clcd.dat'
            out2 = path+'/clcd_wall.dat'

        if os.path.exists(out1):
            out = out1
        elif os.path.exists(out2): 
            out = out2
        else:
            return False, CL, CD, Cm, CDp, CDf

        CLs = np.zeros(n)
        CDs = np.zeros(n)
        Cms = np.zeros(n)
        CDps = np.zeros(n)
        CDfs = np.zeros(n)
        with open(out, 'r') as f:
            lines = f.readlines()
            n_all = len(lines)

            i = 1
            k = 0
            while i<n_all-4 and k<n:

                L1 = lines[-i].split()
                L2 = lines[-i-1].split()
                i += 1

                if L1[2] == L2[2]:
                    # Duplicated lines of the final step when using multiple blocks
                    continue

                CLs[k] = float(L1[5])
                CDs[k] = float(L1[6])
                Cms[k] = float(L1[12])
                CDps[k] = float(L1[8])
                CDfs[k] = float(L1[9])
                k += 1

        CL_ = np.mean(CLs)
        if np.std(CL) < max(0.01, 0.01*CL_):
            CL = CL_
            CD = np.mean(CDs)
            Cm = np.mean(Cms)
            CDp = np.mean(CDps)
            CDf = np.mean(CDfs)

        else:
            converge = False

        return converge, CL, CD, Cm, CDp, CDf

    @staticmethod
    def readAoA(path: str, n=10):
        '''
        Read cfl3d.alpha of the CFL3D outputs. \n
            path:   folder that contains the results
            n:      get the mean value of final n steps

        Return: \n
            succeed (bool), AoA
        '''
        succeed = True
        AoA = 0.0

        if platform.system() in 'Windows':
            out = path+'\\cfl3d.alpha'
        else:
            out = path+'/cfl3d.alpha'

        if not os.path.exists(out):
            return False, AoA

        AoAs = np.zeros(n)
        with open(out, 'r') as f:
            lines = f.readlines()

            if len(lines)==0:
                f.close()
                return False, AoA

            for k in range(n):
                L1 = lines[-k-1].split()
                AoAs[k] = float(L1[3])

        AoA = np.mean(AoAs)

        return succeed, AoA

    @staticmethod
    def readinput(path: str):
        '''
        Read cfl3d.inp of the CFL3D input. \n
            path:   folder that contains the input files

        Return: \n
            succeed (bool), Minf, AoA0 (deg), Re (e6, /m), l2D(bool)
        '''

        succeed = True
        Minf = 0.0
        AoA0 = 0.0
        Re = 0.0
        l2D = False

        if platform.system() in 'Windows':
            inp = path+'\\cfl3d.inp'
        else:
            inp = path+'/cfl3d.inp'

        if not os.path.exists(inp):
            return False, Minf, AoA0, Re, l2D

        with open(inp, 'r') as f:
            lines = f.readlines()

            for i in range(len(lines)-1):
                line = lines[i].split()

                if 'XMACH' in line[0]:
                    L1 = lines[i+1].split()
                    Minf = float(L1[0])
                    AoA0 = float(L1[1])
                    Re   = float(L1[3])

                if 'NGRID' in line[0]:
                    L1 = lines[i+1].split()
                    if int(L1[5])==1:
                        l2D = True

        return succeed, Minf, AoA0, Re, l2D

    @staticmethod
    def readprt(path: str, fname='cfl3d.prt'):
        '''
        Read cfl3d.prt of the CFL3D output. \n
            path:   folder that contains the output files.

        Return: \n
            succeed (bool)
        '''
        mi = 10000000   # maximum size of i*j*k
        ijk = np.zeros([mi,3],  dtype=int)
        xyz = np.zeros([mi,11], dtype=float)

        if platform.system() in 'Windows':
            prt  = path+'\\'+fname
            out1 = path+'\\surface.dat'
            out2 = path+'\\surface2.dat'
        else:
            prt  = path+'/'+fname
            out1 = path+'/surface.dat'
            out2 = path+'/surface2.dat'

        if not os.path.exists(prt):
            return False

        f0 = open(prt, 'r')
        f1 = None
        f2 = None

        block_p = 0
        block_v = 0
        while True:

            line = f0.readline()
            if line == '':
                break

            line = line.split()
            if len(line) == 0:
                continue
            if not line[0] in 'I':
                continue

            if line[6] in 'U/Uinf':
                #* Pressure distribution
                imax = 0
                jmax = 0
                kmax = 0
                imin = mi
                jmin = mi
                kmin = mi
                i0 = 0
                while True:
                    L1 = f0.readline()
                    L1 = L1.split()
                    if len(L1) == 0:
                        break
                    if L1[0] in 'I':
                        break
                    for i in range(3):
                        ijk[i0, i] = int(L1[i])
                    for i in range(11):
                        xyz[i0, i] = float(L1[i+3])
                    
                    imax = max(imax, ijk[i0,0])
                    jmax = max(jmax, ijk[i0,1])
                    kmax = max(kmax, ijk[i0,2])
                    imin = min(imin, ijk[i0,0])
                    jmin = min(jmin, ijk[i0,1])
                    kmin = min(kmin, ijk[i0,2])
                    i0 += 1
                
                nn = (imax-imin+1)*(jmax-jmin+1)*(kmax-kmin+1)
                if i0 == nn:

                    if block_p==0:
                        f1 = open(out1, 'w')
                        f1.write('Variables = X Y Z I J K U V W P T M Cp ut \n')

                    block_p += 1
                    print('  Read pressure block %d'%(block_p))

                    if imax==imin:
                        f1.write('zone T="%d" i= %d j= %d k= %d \n'%(block_p, imax-imin+1, jmax-jmin+1, kmax-kmin+1))
                    elif jmax==jmin:
                        f1.write('zone T="%d" i= %d j= %d k= %d \n'%(block_p, kmax-kmin+1, jmax-jmin+1, imax-imin+1))
                    else:
                        f1.write('zone T="%d" i= %d j= %d k= %d \n'%(block_p, jmax-jmin+1, imax-imin+1, kmax-kmin+1))
                    for i in range(nn):
                        L2 = '%18.10f %18.10f %18.10f'%(xyz[i,0], xyz[i,1], xyz[i,2])
                        L2 = L2 + ' %5d %5d %5d'%(ijk[i,0], ijk[i,1], ijk[i,2])
                        for j in range(8):
                            L2 = L2 + ' %18.10f'%(xyz[i,3+j])
                        f1.write(L2+'\n')

            if line[6] in 'dn':
                #* Viscous distribution

                imax = 0
                jmax = 0
                kmax = 0
                imin = mi
                jmin = mi
                kmin = mi
                i0 = 0
                while True:
                    L1 = f0.readline()
                    L1 = L1.split()
                    if len(L1) == 0:
                        break
                    if L1[0] in 'I':
                        break

                    for i in range(3):
                        ijk[i0, i] = int(L1[i])
                    for i in range(9):
                        xyz[i0, i] = float(L1[i+3])
                    
                    imax = max(imax, ijk[i0,0])
                    jmax = max(jmax, ijk[i0,1])
                    kmax = max(kmax, ijk[i0,2])
                    imin = min(imin, ijk[i0,0])
                    jmin = min(jmin, ijk[i0,1])
                    kmin = min(kmin, ijk[i0,2])
                    i0 += 1
                
                nn = (imax-imin+1)*(jmax-jmin+1)*(kmax-kmin+1)
                if i0 == nn:

                    if block_v==0:
                        f2 = open(out2, 'w')
                        f2.write('Variables = X Y Z I J K dn P T Cf Ch yplus \n')

                    if imax==imin:
                        f2.write('zone i= %d j= %d k= %d \n'%(imax-imin+1, jmax-jmin+1, kmax-kmin+1))
                    elif jmax==jmin:
                        f2.write('zone i= %d j= %d k= %d \n'%(kmax-kmin+1, jmax-jmin+1, imax-imin+1))
                    else:
                        f2.write('zone i= %d j= %d k= %d \n'%(jmax-jmin+1, imax-imin+1, kmax-kmin+1))
                    for i in range(nn):
                        L2 = '%18.10f %18.10f %18.10f'%(xyz[i,0], xyz[i,1], xyz[i,2])
                        L2 = L2 + ' %5d %5d %5d'%(ijk[i,0], ijk[i,1], ijk[i,2])
                        for j in range(6):
                            L2 = L2 + ' %18.10f'%(xyz[i,3+j])
                        f2.write(L2+'\n')

                    block_v += 1
                    print('  Read viscous block %d'%(block_v))

        f0.close()
        if f1 is not None:
            f1.close()
        if f2 is not None:
            f2.close()

        return True

    @staticmethod
    def readprt_foil(path: str, j0: int, j1: int, fname='cfl3d.prt'):
        '''
        Read and extract foil Cp from cfl3d.prt \n
            path:   folder that contains the output files.
            j0:     j index of the lower surface TE
            j1:     j index of the upper surface TE

        cfl3d.prt index \n
            i : 1 - 1   symmetry plane
            j : 1 - nj  from far field of lower surface TE to far field of upper surface TE
            k : 1 - nk  from surface to far field

        Return: \n
            succeed (bool), (field: X,Y,U,V,P,T,Ma,Cp,vi), (foil: x, y, Cp)
        '''

        if platform.system() in 'Windows':
            prt  = path+'\\'+fname
        else:
            prt  = path+'/'+fname

        if not os.path.exists(prt):
            return False, None, None

        X = None

        f0 = open(prt, 'r')
        while True:

            line = f0.readline()
            if line == '':
                break

            line = line.split()
            if len(line) == 0:
                continue

            if 'BLOCK' in line[0]:
                ni = int(line[-3])
                nj = int(line[-2])
                nk = int(line[-1])

                X = np.zeros([nj, nk])
                Y = np.zeros([nj, nk])
                U = np.zeros([nj, nk])
                V = np.zeros([nj, nk])
                P = np.zeros([nj, nk])
                T = np.zeros([nj, nk])
                Ma = np.zeros([nj, nk])
                Cp = np.zeros([nj, nk])
                vi = np.zeros([nj, nk])
                continue

            if not line[0] in 'I':
                continue

            for k in range(nk):
                for j in range(nj):
                    L1 = f0.readline()
                    L1 = L1.split()

                    X [j,k] = float(L1[3])
                    Y [j,k] = float(L1[4])
                    U [j,k] = float(L1[6])
                    V [j,k] = float(L1[7])
                    P [j,k] = float(L1[9])
                    T [j,k] = float(L1[10])
                    Ma[j,k] = float(L1[11])
                    Cp[j,k] = float(L1[12])
                    vi[j,k] = float(L1[13])

            break

        if X is None:
            return False, None, None

        field = (X,Y,U,V,P,T,Ma,Cp,vi)
        f0.close()

        nn = j1-j0
        foil_x = np.zeros(nn)
        foil_y = np.zeros(nn)
        foil_Cp = np.zeros(nn)

        for i in range(nn):
            foil_x[i]  = X[j0+i,0]
            foil_y[i]  = Y[j0+i,0]
            foil_Cp[i] = Cp[j0+i,0]

        foil = (foil_x, foil_y, foil_Cp)

        return True, field, foil








