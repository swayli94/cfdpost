'''
This module contains a class to extract flow features of airfoils or wing sections.
'''
import numpy as np
import copy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class FeatureSec():
    '''
    Extracting flow features
    '''
    _i  = 0     # index of the mesh point
    _X  = 0.001 # location of the feature location
    _value = 0.0

    #* Dictionary of flow features (identify the index and location)
    xf_dict = {
        'Cu':  ['upper crest', _i, _X],             # crest point on upper surface
        'Cl':  ['lower crest', _i, _X],             # crest point on lower surface
        'tu':  ['upper highest', _i, _X],           # highest point on upper surface
        'tl':  ['lower highest', _i, _X],           # lowest point on lower surface
        'tm':  ['max thickness', _i, _X],           # maximum thickness position

        'L': ['upper LE', _i, _X],                  # suction peak near leading edge on upper surface
        'T': ['upper TE', _i, _X],                  # trailing edge upper surface (98% chord length)
        'S': ['separation start', _i, _X],          # separation start position
        'R': ['reattachment', _i, _X],              # reattachment position
        'Q': ['lower LE', _i, _X],                  # suction peak near leading edge on lower surface
        'M': ['lower surface max Ma', _i, _X],      # position of lower surface maximum Mach number
        'mUy': ['min(du/dy)', _i, _X],              # position of min(du/dy)

        'F': ['shock foot', _i, _X],                # shock foot position
        '1': ['shock front', _i, _X],               # shock wave front position
        '3': ['shock hind', _i, _X],                # position of just downstream the shock
        'U': ['local sonic', _i, _X],               # local sonic position
                                                    # Note: for weak shock waves, may not reach Mw=1
                                                    #       define position of U as Mw minimal extreme point after shock foot
        'N': ['new flat boundary', _i, _X],         # starting position of new flat boundary
        'Hi':  ['maximum Hi', _i, _X],              # position of maximum Hi
        'Hc':  ['maximum Hc', _i, _X],              # position of maximum Hc
        
        'L12': ['length 1~2', _value],              # X2-X1
        'L1U': ['length 1~U', _value],              # XU-X1
        'L13': ['length 1~3', _value],              # X3-X1
        'LSR': ['length S~R', _value],              # XR-XS
        'lSW': ['single shock', _value],            # single shock wave flag
        'DCp': ['shock strength', _value],          # Cp change through shock wave
        'Err': ['suc Cp area', _value],             # Cp integral of suction plateau fluctuation
        'CLU': ['upper CL', _value],                # CL of upper surface
        'AFL': ['aft loading', _value],             # CL of aft loading
        'kaf': ['slope aft', _value]                # average Mw slope of the aft upper surface (N~T)
    }

    def __init__(self, Minf, AoA, Re):
        '''
        Minf:       Free stream Mach number
        AoA:        Angle of attack (deg)
        Re:         Reynolds number per meter
        '''
        self.Minf = Minf
        self.AoA  = AoA
        self.Re   = Re
        self.xf_dict = copy.deepcopy(FeatureSec.xf_dict)

    def setdata(self, x, y, Cp, Tw, Hi, Hc, dudy):
        '''
        Set the data of this foil or section.   \n
            Data:   ndarray, start from lower surface trailing edge
        '''
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.Cp = copy.deepcopy(Cp)
        self.Mw = self.Cp2Mw()
        self.Tw = copy.deepcopy(Tw)
        self.Hi = copy.deepcopy(Hi)
        self.Hc = copy.deepcopy(Hc)
        self.dudy = copy.deepcopy(dudy)

        iLE = np.argmin(self.x)
        self.x -= self.x[iLE]
        self.y -= self.y[iLE]
        self.x[0] = 1.0
        self.x[-1] = 1.0

        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')
        fhu = interp1d(self.x[iLE:], self.Hc[iLE:], kind='cubic')
        gu  = interp1d(self.x[iLE:], self.y [iLE:], kind='cubic')
        x_  = np.append(self.x[iLE:0:-1], self.x[0])
        y_  = np.append(self.y[iLE:0:-1], self.y[0])
        gl  = interp1d(x_, y_, kind='cubic')

        self.xx = np.arange(0.0, 1.0, 0.001)
        self.yu = gu(self.xx)
        self.yl = gl(self.xx)
        self.mu = fmw(self.xx)
        self.hu = fhu(self.xx)
        
        self.iLE = iLE

    @property
    def n_point(self):
        '''
        Number of points in this section
        '''
        return self.x.shape[0]

    @staticmethod
    def IsentropicCp(Ma, Minf: float, g=1.4):
        ''' 
        Isentropic flow: Calculate Cp by Mach \n
            Ma:     float, or ndarray
            Minf:   free stream Mach number
            g:      γ=1.4, ratio of the specific heats
        '''
        X = (2.0+(g-1.0)*Minf**2)/(2.0+(g-1.0)*Ma**2)
        X = X**(g/(g-1.0))
        Cp = 2.0/g/Minf**2*(X-1.0)

        return Cp

    def Cp2Mw(self, n_ref=100, M_max=2.0):
        '''
        Converting Cp to wall Mach number
        '''
        Ma_ref = np.linspace(0.0, M_max, n_ref)
        Cp_ref = self.IsentropicCp(Ma_ref, self.Minf)

        f = interp1d(Cp_ref, Ma_ref, kind='cubic')

        return f(self.Cp)

    @staticmethod
    def ShapeFactor(sS, VtS, Tw: float, iUe: int):
        '''
        Calculate shape factor Hi & Hc by mesh points on a line pertenticular to the wall. \n
            sS:     ndarray (n), distance of mesh points to wall
            VtS:    ndarray (n), velocity component of mesh points (parallel to the wall)
            Tw:     wall temperature (K)
            iUe:    index of mesh point locating the outer velocity Ue

        Return: \n
            Hi:     incompressible shape factor
            Hc:     compressible shape factor

        Note: \n
            XR  => 物面参考点，考察以 XR 为起点，物面法向 nR 方向上的数据点，共 nHi 个数据点
            sS  => 数据点到物面距离
            VtS => 数据点速度在物面方向的分量

            se:     distance of boundary layer outer boundary to wall
            ds:     𝛿*, displacement thickness
            tt:     θ, momentum loss thickness
            Ue:     outer layer velocity component (parallel to the wall)
            Ue      测试结果显示，直接取最大Ue较为合理，取一定范围内平均，或取固定网格的值，效果不好
        '''
        iUe = min(sS.shape[0], iUe)
        Ue = VtS[iUe]
        se = sS[iUe]
        ds = 0.0
        tt = 0.0

        for i in range(iUe-1):
            a1 = Ue-VtS[i]
            a2 = Ue-VtS[i+1]
            ds += 0.5*(a1+a2)*(sS[i+1]-sS[i])

        for i in range(iUe-1):
            a1 = VtS[i  ]*(Ue-VtS[i  ])
            a2 = VtS[i+1]*(Ue-VtS[i+1])
            tt += 0.5*(a1+a2)*(sS[i+1]-sS[i])

        Hi = ds/tt*Ue
        Hc = Tw*Hi+Tw-1

        return Hi, Hc

    @staticmethod
    def getHi(X, Y, U, V, T, j0: int, j1: int, nHi: int):
        '''
        Calculate shape factor Hi & Hc from field data \n
            Field data: ndarray (nj,nk), X, Y, U, V, T
            j0:     j index of the lower surface TE
            j1:     j index of the upper surface TE
            nHi:    maximum number of mesh points in k direction for boundary layer

        Return: \n
            Hi, Hc: ndarray (j1-j0)
            info:   tuple of ndarray (Tw, dudy)

        Note: \n
            Tw:     wall temperature
            dudy:   du/dy
            iUe:    index of mesh point locating the outer velocity Ue
            XR:     reference position on the wall

        Filed data (j,k) index \n
            j: 1  - nj  from far field of lower surface TE to far field of upper surface TE
            j: j0 - j1  from lower surface TE to upper surface TE
            k: 1  - nk  from surface to far field (assuming pertenticular to the wall)
        '''

        iLE = int(0.5*(j0+j1))
        nj = X.shape[0]
        nk = X.shape[1]
        nn = j1-j0

        Hi = np.zeros(nn)
        Hc = np.zeros(nn)
        Tw = np.zeros(nn)
        dudy = np.zeros(nn)

        #* Locate boundary layer edge index iUe & calculate du/dy
        sS  = np.zeros([nn,nHi])
        VtS = np.zeros([nn,nHi])
        iUe = np.zeros(nn, dtype=int)

        for j in range(nn):
            jj = j0+j
            XR = np.array([X[jj,0], Y[jj,0]])
            tR = np.array([X[jj+1,0]-X[jj-1,0], Y[jj+1,0]-Y[jj-1,0]])
            tR = tR/np.linalg.norm(tR)
            if tR[0]<0.0:
                tR = -tR

            for i in range(nHi-1):
                XS = np.array([X[jj,i+1], Y[jj,i+1]])
                VS = np.array([U[jj,i+1], V[jj,i+1]])

                sS [j,i+1] = np.linalg.norm(XR-XS)
                VtS[j,i+1] = np.dot(tR,VS)

            iUe[j]  = np.argmax(np.abs(VtS[j,:]))
            dudy[j] = VtS[j,1]/sS[j,1]
            Tw[j]   = T[jj,0]
        
        #* Smooth iUe at shock wave foot
        nspan = 4
        for j in range(nn-2*nspan):
            jj = j+nspan
            r1 = 0.5*(iUe[jj-nspan]+iUe[jj+nspan])
            r2 = abs(iUe[jj+nspan]-iUe[jj-nspan])
            r3 = abs(iUe[jj]-iUe[jj-nspan]) + abs(iUe[jj]-iUe[jj+nspan])
            if r3>r2:
                iUe[jj] = int(r1)

        #* Calculate Hi & Hc
        for j in range(nn):
            Hi[j], Hc[j] = FeatureSec.ShapeFactor(sS[j,:], VtS[j,:], Tw[j], iUe[j])

        #* Limit leading edge Hi
        r1 = 1.0
        r2 = 1.0
        r3 = 1.0
        r4 = 1.0
        for j in range(nn):
            jj = j0+j
            if (X[jj,0]-0.05)*(X[jj+1,0]-0.05)<=0.0 and jj<iLE:
                r1 = Hi[j]
                r3 = Hc[j]
            if (X[jj,0]-0.05)*(X[jj+1,0]-0.05)<=0.0 and jj>=iLE:
                r2 = Hi[j]
                r4 = Hc[j]

        for j in range(nn):
            jj = j0+j
            if X[jj,0]<0.05 and jj<iLE:
                Hi[j] = r1
                Hc[j] = r3
            if X[jj,0]<0.05 and jj>=iLE:
                Hi[j] = r2
                Hc[j] = r4

        return Hi, Hc, (Tw, dudy)

    def getValue(self, feature: str, key: str):
        '''
        Get value of given feature. \n
            feature:    key of feature dictionary
            key:        'i', 'X', 'Cp', 'Mw', 'Tw', 'Hi', 'Hc', 'dudy'
        '''

        if not feature in FeatureSec.xf_dict.keys:
            print('  Warning: feature [%s] not valid'%(feature))
            return 0.0

        aa = self.xf_dict[feature]

        if len(aa)==2:
            return aa[1]

        if key in 'i':
            return aa[1]

        if key in 'X':
            return aa[2]

        if key in 'Cp':
            yy = self.Cp
        elif key in 'Mw':
            yy = self.Mw
        elif key in 'Tw':
            yy = self.Tw
        elif key in 'Hi':
            yy = self.Hi
        elif key in 'Hc':
            yy = self.Hc
        elif key in 'dudy':
            yy = self.dudy
        else:
            raise Exception('  key %s not valid'%(key))

        ii = aa[1]
        xx = aa[2]
        X = self.x[ii-2:ii+3]
        Y = yy[ii-2:ii+3]
        f = interp1d(X, Y, kind='cubic')

        return f(xx)

    #TODO: locate the position of flow features
    def locate_basic(self):
        '''
        Locate the index and position of basic flow features.

        Basic: L, T, Q, M, S, R, mUy
        '''
        X = self.x
        M = self.Mw
        dudy = self.dudy

        nn  = X.shape[0]
        iLE = self.iLE

        #TODO: Basic features
        #* L => suction peak near leading edge on upper surface
        for i in range(int(0.2*nn)):
            ii = i + iLE
            if M[ii-1]<=M[ii] and M[ii]>=M[ii+1]:
                self.xf_dict['L'][1] = ii
                self.xf_dict['L'][2] = X[ii]
                break

        #* T => trailing edge upper surface (98% chord length)
        for i in range(int(0.2*nn)):
            ii = nn-i-1
            if X[ii]<=0.98 and X[ii+1]>0.98:
                self.xf_dict['T'][1] = ii
                self.xf_dict['T'][2] = 0.98
                break
        
        #* Q => suction peak near leading edge on lower surface
        for i in range(int(0.2*nn)):
            ii = iLE - i
            if M[ii-1]<=M[ii] and M[ii]>=M[ii+1]:
                self.xf_dict['Q'][1] = ii
                self.xf_dict['Q'][2] = X[ii]
                break

        #* M => position of lower surface maximum Mach number
        ii = np.argmax(M[:iLE])
        self.xf_dict['M'][1] = ii
        self.xf_dict['M'][2] = X[ii]

        #* S => separation start position
        #* R => reattachment position
        #* mUy => position of min(du/dy)
        min_Uy = 1e6
        for i in range(int(0.5*nn)):
            ii = iLE + i
            if X[ii]<0.2:
                continue
            if X[ii]>0.98:
                break

            if dudy[ii]>=0.0 and dudy[ii+1]<0.0:
                self.xf_dict['S'][1] = ii
                self.xf_dict['S'][2] = (0.0-dudy[ii])/(dudy[ii+1]-dudy[ii])

            if dudy[ii]<=0.0 and self.dudy[ii+1]>0.0:
                self.xf_dict['R'][1] = ii
                self.xf_dict['R'][2] = (0.0-dudy[ii])/(dudy[ii+1]-dudy[ii])
        
            if dudy[ii]<min_Uy and dudy[ii-1]>=dudy[ii] and dudy[ii+1]>=dudy[ii]:
                min_Uy = dudy[ii]
                self.xf_dict['mUy'][1] = ii
                self.xf_dict['mUy'][2] = X[ii]

    def locate_geo(self):
        '''
        Locate the index and position of geometry related flow features.\n
            AoA:    deg

        Geo: Cu, Cl, tu, tl, tm
        '''
        X  = self.x
        xx = self.xx
        yu = self.yu
        yl = self.yl
        iLE = self.iLE

        #* tm => maximum thickness
        #* tu => highest point on upper surface
        #* tl => lowest point on lower surface
        x_max = xx[np.argmax(yu-yl)]
        x_mu  = xx[np.argmax(yu)]
        x_ml  = xx[np.argmin(yl)]

        self.xf_dict['tm'][1] = np.argmin(np.abs(X[iLE:]-x_max)) + iLE
        self.xf_dict['tm'][2] = x_max
        self.xf_dict['tu'][1] = np.argmin(np.abs(X[iLE:]-x_mu )) + iLE
        self.xf_dict['tu'][2] = x_mu
        self.xf_dict['tl'][1] = np.argmin(np.abs(X[:iLE]-x_ml ))
        self.xf_dict['tl'][2] = x_ml

        #* Cu => crest point on upper surface
        aa = self.AoA/180.0*np.pi
        x0 = np.array([0.0, 0.0])
        x1 = np.array([np.cos(aa), np.sin(aa)])
        rr = -1.0
        rc = 0.0
        xc1= 0.0
        for i in range(xx.shape[0]):
            xt = np.array([xx[i], yu[i]])
            s, _ = ratio_vec(x0, x1, xt)
            if s < rr:
                break
            if s > rc:
                rc = s
                xc1 = xx[i]
            rr = s

        self.xf_dict['Cu'][1] = np.argmin(np.abs(X[iLE:]-xc1)) + iLE
        self.xf_dict['Cu'][2] = xc1

        #* Cl => crest point on lower surface
        rr = -1.0
        rc = 0.0
        xc2= 0.0
        for i in range(xx.shape[0]):
            xt = np.array([xx[i], yl[i]])
            s, _ = ratio_vec(x0, x1, xt)
            if s < rr:
                break
            if s > rc:
                rc = s
                xc2 = xx[i]
            rr = s

        self.xf_dict['Cl'][1] = np.argmin(np.abs(X[:iLE]-xc2))
        self.xf_dict['Cl'][2] = xc2

    def locate_shock(self, dMwcri_1=-1.0):
        '''
        Locate the index and position of shock wave related flow features. \n
            dMwcri_1: critical value locating shock wave front

        Shock: 1, 3, F, U, N, Hi, Hc
        '''
        X   = self.x
        xx  = self.xx
        mu  = self.mu
        hu  = self.hu
        nn  = xx.shape[0]
        iLE = self.iLE

        dMw = np.zeros(nn)
        for i in range(nn-1):
            if xx[i]>0.98:
                continue
            dMw[i] = (mu[i+1]-mu[i])/(xx[i+1]-xx[i])
            dMw[i] = min(dMw[i], 2)

        flag = FeatureSec.check_singleshock(xx, mu, dMw)
        if not flag==1:
            return

        #* F => shock foot position
        i_F = np.argmin(dMw)
        x_F = xx[i_F]
        self.xf_dict['F'][1] = np.argmin(np.abs(X[iLE:]-x_F)) + iLE
        self.xf_dict['F'][2] = x_F

        #* 1 => shock wave front position
        # Find the kink position of dMw in range [x_F-0.2, x_F], defined as dMw = -1
        i_1 = 0
        for i in np.arange(i_F, 1, -1):
            if xx[i]<x_F-0.2:
                break
            if dMw[i]>=dMwcri_1 and dMw[i+1]<dMwcri_1:
                i_1 = i
                break
        x_1 = xx[i_1]
        self.xf_dict['1'][1] = np.argmin(np.abs(X[iLE:]-x_1)) + iLE
        self.xf_dict['1'][2] = x_1

        #* 3 => position of just downstream the shock
        # Find the first flat position of Mw in range [x_F+0.2, x_F], defined as dMw = 0 or -1
        i_3 = 0
        for i in np.arange(i_F, nn-1, 1):
            if xx[i]>x_F+0.2:
                break
            if dMw[i]<=dMwcri_1 and dMw[i+1]>dMwcri_1:
                i_3 = i
            if dMw[i]<=0.0 and dMw[i+1]>0.0:
                i_3 = i
                break

        x_3 = xx[i_3]
        self.xf_dict['3'][1] = np.argmin(np.abs(X[iLE:]-x_3)) + iLE
        self.xf_dict['3'][2] = x_3

        #* U => local sonic position
        i_U = 0
        for i in np.arange(i_1, i_3, 1):
            if mu[i]>=1.0 and mu[i+1]<1.0:
                i_U = i
                break
        x_U = xx[i_U]
        self.xf_dict['U'][1] = np.argmin(np.abs(X[iLE:]-x_U)) + iLE
        self.xf_dict['U'][2] = x_U

        #* Hi, Hc => position of maximum Hi, Hc after shock wave front
        # For cases when shock wave is weak, and Hc just keeps growing, set 0
        i_H = 0
        max1 = 0.0
        for i in np.arange(i_1, nn-1, 1):
            if hu[i-1]<=hu[i] and hu[i+1]<=hu[i] and hu[i]>max1:
                max1 = hu[i]
                i_H = i

        x_H = xx[i_H]
        self.xf_dict['Hc'][1] = np.argmin(np.abs(X[iLE:]-x_H)) + iLE
        self.xf_dict['Hc'][2] = x_H
        self.xf_dict['Hi'][1] = self.xf_dict['Hc'][1]
        self.xf_dict['Hi'][2] = x_H

        #* N => starting position of new flat boundary
        # i.e., position of minimum Hc after shock wave front
        # For cases when shock wave is weak, and Hc just keeps growing, set 0
        i_N = 0
        min1 = 1000.0
        for i in np.arange(i_1, nn-1, 1):
            if hu[i-1]>=hu[i] and hu[i+1]<=hu[i] and hu[i]<min1:
                min1 = hu[i]
                i_N = i

        x_N = xx[i_N]
        self.xf_dict['N'][1] = np.argmin(np.abs(X[iLE:]-x_N)) + iLE
        self.xf_dict['N'][2] = x_N

    @staticmethod
    def check_singleshock(xu, Mw, dMw, dMwcri_F=-2.0):
        '''
        Check whether is single shock wave or not \n
            xx:     ndarray, x location
            Mw:     ndarray, wall Mach number
            dMw:    ndarray, slope of wall Mach number
            dMwcri_F: critical value filtering shock wave

        Return: flag\n
            1:  single shock wave
            0:  shockless
            -1: multiple shock waves 
        '''
        nn = xu.shape[0]
        dm = dMw.copy()
        i1 = np.argmin(dm)
        d1 = dm[i1]
        
        # Check if shockless
        if Mw[i1]<1.0 or dm[i1]>dMwcri_F:
            return 0

        # Check if second shock wave exists
        for i in np.arange(i1, nn, 1, dtype=int):
            if dm[i]<=0.0:
                dm[i]=0.0
            else:
                break

        for i in np.arange(i1, 0, -1, dtype=int):
            if dm[i]<=0.0:
                dm[i]=0.0
            else:
                break
        
        i2 = np.argmin(dm)
        if Mw[i1]>1.0 and dm[i1]<max(dMwcri_F, 0.5*d1):
            return -1

        return 1

    def aux_features(self):
        '''
        Calculate auxiliary features based on basic, geo, and shock features.

        Length, lSW, DCp, Err, CLU, AFL, kaf
        '''



    def extract_features(self):
        '''
        Extract flow features list in the dictionart.
        '''
        self.locate_basic()
        self.locate_geo()
        self.locate_shock()
        self.aux_features()


def ratio_vec(x0, x1, x):
    '''
    Calculate distance s to vector x1-x0.   \n
        x0, x1: ndarray, start and end point of the vector
        x:      ndarray, current point

    Return: \n
        s:  distance to line
        t:  ratio of (projected |x0x|) / |x0x1|
    '''
    l0 = np.linalg.norm(x0-x1) + 1e-20
    l1 = np.linalg.norm(x0-x ) + 1e-20
    v  = (x1-x0)/l0
    l2 = np.dot(v, x-x0)
    t  = l2/l1
    s  = np.sqrt(l1**2 - l2**2)

    return s, t

def curve_curvature(x, y):
    '''
    Calculate curvature of points in the curve
        x, y: points of curve (list or ndarray)

    Return: curv ndarray
    '''
    nn = len(x)
    if nn<3:
        raise Exception('curvature needs at least 3 points')
    
    curv = np.zeros(nn)
    for i in range(1, nn-1):
        X1 = np.array([x[i-1], y[i-1]])
        X2 = np.array([x[i  ], y[i  ]])
        X3 = np.array([x[i+1], y[i+1]])

        a = np.linalg.norm(X1-X2)
        b = np.linalg.norm(X2-X3)
        c = np.linalg.norm(X3-X1)
        p = 0.5*(a+b+c)
        t = p*(p-a)*(p-b)*(p-c)
        R = a*b*c
        if R <= 1.0E-12:
            curv_ = 0.0
        else:
            curv_ = 4.0*np.sqrt(t)/R

        a1 = X2[0] - X1[0]
        a2 = X2[1] - X1[1]
        b1 = X3[0] - X1[0]
        b2 = X3[1] - X1[1]
        if a1*b2 < a2*b1:
            curv_ = -curv_

        curv[i] = curv_

    curv[0] = curv[1]
    curv[-1] = curv[-2]

    return curv