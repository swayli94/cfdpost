'''
Extract physical features of airfoils or wing sections.
'''
import copy
import os

import numpy as np
from scipy.interpolate import interp1d


class PhysicalSec():
    '''
    Extracting flow features of a section (features on/near the wall)
    '''
    _i  = 0     # index of the mesh point
    _X  = 0.0   # location of the feature location
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
        'D': ['dent on plateau', _i, _X],           # dent on the suction plateau
        'U': ['local sonic', _i, _X],               # local sonic position
                                                    # Note: for weak shock waves, may not reach Mw=1
                                                    #       define position of U as Mw minimal extreme point after shock foot
        'A': ['maximum Mw after shock', _i, _X],    # maximum wall Mach number after shock wave (or equal to 3)
        'N': ['new flat boundary', _i, _X],         # starting position of new flat boundary
                                                    # most of the time, A == N
        'Hi':  ['maximum Hi', _i, _X],              # position of maximum Hi
        'Hc':  ['maximum Hc', _i, _X],              # position of maximum Hc
        
        'L1U': ['length 1~U', _value],              # XU-X1
        'L13': ['length 1~3', _value],              # X3-X1
        'LSR': ['length S~R', _value],              # XR-XS
        'lSW': ['single shock', _value],            # single shock wave flag
        'DCp': ['shock strength', _value],          # Cp change through shock wave
        'Err': ['suc Cp area', _value],             # Cp integral of suction plateau fluctuation
        'DMp': ['Mw dent on plateau', _value],      # Mw dent on suction plateau
        'CLU': ['upper CL', _value],                # CL of upper surface
        'kaf': ['slope aft', _value]                # average Mw slope of the aft upper surface (3/N~T)
    }

    def __init__(self, Minf, AoA, Re):
        '''
        ### Inputs:
        ```text
        Minf:       Free stream Mach number
        AoA:        Angle of attack (deg)
        Re:         Reynolds number per meter
        ```
        '''
        self.Minf = Minf
        self.AoA  = AoA
        self.Re   = Re
        self.xf_dict = copy.deepcopy(PhysicalSec.xf_dict)

    def setdata(self, x, y, Cp, Tw, Hi, Hc, dudy):
        '''
        Set the data of this foil or section.

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

    def set_Mw(self, x, Mw):
        '''
        Set the Mw distribution of this foil or section.

        Data:   ndarray, start from lower surface trailing edge
        '''
        self.x  = copy.deepcopy(x)
        self.Mw = copy.deepcopy(Mw)

        iLE = np.argmin(self.x)
        self.iLE = iLE

        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')
        self.xx = np.arange(0.0, 1.0, 0.001)
        self.mu = fmw(self.xx)

    @property
    def n_point(self):
        '''
        Number of points in this section
        '''
        return self.x.shape[0]

    @staticmethod
    def IsentropicCp(Ma, Minf: float, g=1.4):
        ''' 
        Isentropic flow: Calculate Cp by Mach

        ### Inputs:
        ```text
        Ma:     float, or ndarray
        Minf:   free stream Mach number
        g:      γ=1.4, ratio of the specific heats
        ```
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

        Cp_ = self.Cp.copy()
        Cp_ = np.clip(Cp_, Cp_ref[-1], Cp_ref[0])

        return f(Cp_)

    @staticmethod
    def ShapeFactor(sS, VtS, Tw: float, iUe: int):
        '''
        Calculate shape factor Hi & Hc by mesh points on a line pertenticular to the wall.

        ### Inputs:
        ```text
        sS:     ndarray (n), distance of mesh points to wall
        VtS:    ndarray (n), velocity component of mesh points (parallel to the wall)
        Tw:     wall temperature (K)
        iUe:    index of mesh point locating the outer velocity Ue
        ```

        ### Return:
        ```text
        Hi:     incompressible shape factor
        Hc:     compressible shape factor
        ```

        ### Note:
        ```text
        XR  => 物面参考点，考察以 XR 为起点，物面法向 nR 方向上的数据点，共 nHi 个数据点
        sS  => 数据点到物面距离
        VtS => 数据点速度在物面方向的分量

        se:     distance of boundary layer outer boundary to wall
        ds:     𝛿*, displacement thickness
        tt:     θ, momentum loss thickness
        Ue:     outer layer velocity component (parallel to the wall)
        Ue      测试结果显示，直接取最大Ue较为合理，取一定范围内平均，或取固定网格的值，效果不好
        ```
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
        Calculate shape factor Hi & Hc from field data

        ### Inputs:
        ```text
        Field data: ndarray (nj,nk), X, Y, U, V, T
        j0:     j index of the lower surface TE
        j1:     j index of the upper surface TE
        nHi:    maximum number of mesh points in k direction for boundary layer
        ```

        ### Return:
        ```text
        Hi, Hc: ndarray (j1-j0)
        info:   tuple of ndarray (Tw, dudy)
        ```

        ### Note:
        ```text
        Tw:     wall temperature
        dudy:   du/dy
        iUe:    index of mesh point locating the outer velocity Ue
        XR:     reference position on the wall
        ```

        ### Filed data (j,k) index
        ```text
        j: 1  - nj  from far field of lower surface TE to far field of upper surface TE
        j: j0 - j1  from lower surface TE to upper surface TE
        k: 1  - nk  from surface to far field (assuming pertenticular to the wall)
        ```
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
            Hi[j], Hc[j] = PhysicalSec.ShapeFactor(sS[j,:], VtS[j,:], Tw[j], iUe[j])

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

    def getValue(self, feature: str, key='key') -> float:
        '''
        Get value of given feature.

        ### Inputs:
        ```text
        feature:    key of feature dictionary
        key:        'i', 'X', 'Cp', 'Mw', 'Tw', 'Hi', 'Hc', 'dudy'
        ```
        '''

        if not feature in PhysicalSec.xf_dict.keys():
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

        if xx <= 1e-6:
            return 0.0

        if ii >= self.iLE:
            i0 = max(self.iLE, ii-4)
            i1 = i0 + 7
        else:
            i1 = min(self.iLE, ii+4)
            i0 = i1 - 7

        X = self.x[i0:i1]
        Y = yy[i0:i1]

        f = interp1d(X, Y, kind='cubic')

        return f(xx) 

    #TODO: locate the position of flow features
    def locate_basic(self):
        '''
        Locate the index and position of basic flow features.

        ### Get value of: L, T, Q, M
        '''
        X = self.x
        M = self.Mw

        nn  = X.shape[0]
        iLE = self.iLE

        #TODO: Basic features
        #* L => suction peak near leading edge on upper surface
        # 1: maximum extreme point
        # 2: outermost point in X + Mw = 0 direction
        i_L = 0
        x_L = 0.0
        for i in range(int(0.25*nn)):
            ii = i + iLE
            if M[ii-1]<=M[ii] and M[ii]>=M[ii+1]:
                i_L = ii
                x_L = X[ii]
                break

        if i_L == 0:
            max_L = -1.0e6
            vec = np.array([-1.0,1.0])

            for i in range(int(0.25*nn)):
                ii = i + iLE
                vv = np.array([X[ii], M[ii]])
                dd = np.dot(vv, vec)
                if dd > max_L:
                    max_L = dd
                    i_L = ii
                    x_L = X[ii]

        self.xf_dict['L'][1] = i_L
        self.xf_dict['L'][2] = x_L

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
        i_M = 0
        max1 = -1.0
        for i in np.arange(1, iLE, 1):
            if M[i-1]<=M[i] and M[i+1]<=M[i] and M[i]>max1:
                max1 = M[i]
                i_M = i

        self.xf_dict['M'][1] = i_M
        self.xf_dict['M'][2] = X[i_M]

    def locate_sep(self):
        '''
        Locate the index and position of flow features about du/dy.

        ### Get value of: S, R, mUy
        '''
        X = self.x
        dudy = self.dudy

        nn  = X.shape[0]
        iLE = self.iLE

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

        ### Get value of: Cu, Cl, tu, tl, tm
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
        Locate the index and position of shock wave related flow features.

        ### Get value of: 1, 3, F, U, D, A
        
        ### Inputs:
        ```text
        dMwcri_1: critical value locating shock wave front
        ```
        '''
        X   = self.x
        xx  = self.xx
        mu  = self.mu
        nn  = xx.shape[0]
        iLE = self.iLE

        dMw = np.zeros(nn)
        for i in range(nn-1):
            if xx[i]>0.98:
                continue
            dMw[i] = (mu[i+1]-mu[i])/(xx[i+1]-xx[i])
            dMw[i] = min(dMw[i], 2)

        flag = PhysicalSec.check_singleshock(xx, mu, dMw)
        self.xf_dict['lSW'][1] = flag
        if not flag==1:
            return 0

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
        # Find the first flat position of Mw in range [x_F, x_F+0.2], defined as dMw = 0 or -1
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

        #* D => dent on the suction plateau
        # minimum Mw between L and 1
        x_L = max(self.xf_dict['L'][2], 0.05)
        i_D = 0
        min_D = 10.0
        for i in np.arange(2, i_1-1, 1):

            if xx[i]<x_L:
                continue

            if mu[i-1]>=mu[i] and mu[i]<=mu[i+1] and mu[i]<min_D:
                i_D = i
                min_D = mu[i]

        x_D = xx[i_D]
        self.xf_dict['D'][1] = np.argmin(np.abs(X[iLE:]-x_D)) + iLE
        self.xf_dict['D'][2] = x_D

        #* U => local sonic position
        i_U = 0
        for i in np.arange(i_1, i_3, 1):
            if mu[i]>=1.0 and mu[i+1]<1.0:
                i_U = i
                break
        x_U = xx[i_U]
        self.xf_dict['U'][1] = np.argmin(np.abs(X[iLE:]-x_U)) + iLE
        self.xf_dict['U'][2] = x_U

        #* A => maximum Mw after shock
        # Find the maximum position of Mw in range [x_3, x_3+0.4]
        i_A = 0
        max_A = 0.0
        for i in np.arange(i_3, nn-1, 1):
            if xx[i]>x_3+0.4:
                break
            if mu[i]>max_A:
                i_A = i
                max_A = mu[i]

        x_A = xx[i_A]
        self.xf_dict['A'][1] = np.argmin(np.abs(X[iLE:]-x_A)) + iLE
        self.xf_dict['A'][2] = x_A

        return i_1

    def locate_BL(self, i_1):
        '''
        Locate the index and position of boundary layer related flow features. \n
        
        i-1: index of shock wave front position in self.xx

        ### Get value of: N, Hi, Hc
        '''
        X   = self.x
        xx  = self.xx
        hu  = self.hu
        nn  = xx.shape[0]
        iLE = self.iLE

        #* Hi, Hc => position of maximum Hi, Hc after shock wave front
        # For cases when shock wave is weak, and Hc just keeps growing, set 0
        i_H = 0
        max1 = 0.0
        for i in np.arange(i_1, nn-2, 1):

            if xx[i] > 0.95:
                break

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
        Check whether is single shock wave or not

        ### Inputs:
        ```text
        xx:     ndarray, x location
        Mw:     ndarray, wall Mach number
        dMw:    ndarray, slope of wall Mach number
        dMwcri_F: critical value filtering shock wave
        ```

        ### Return: flag
        ```text
        1:  single shock wave
        0:  shockless
        -1: multiple shock waves 
        ```
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
        if Mw[i2]>1.0 and dm[i2]<max(dMwcri_F, 0.5*d1):
            return -1

        return 1

    def aux_features(self):
        '''
        Calculate auxiliary features based on basic, geo, and shock features.

        ### Get value of: Length, lSW, DCp, Err, DMp, CLU, kaf
        '''
        X  = self.x
        xx = self.xx
        mu = self.mu
        nn = xx.shape[0]
        x1 = self.xf_dict['1'][2]
        
        self.xf_dict['L1U'][1] = self.xf_dict['U'][2] - x1
        self.xf_dict['L13'][1] = self.xf_dict['3'][2] - x1
        self.xf_dict['LSR'][1] = self.xf_dict['R'][2] - self.xf_dict['S'][2]
        self.xf_dict['DCp'][1] = self.getValue('3','Cp') - self.getValue('1','Cp')

        rr = np.cos(self.AoA/180.0*np.pi)
        #* Err => Cp integral of suction plateau fluctuation
        #* DMp => Mw dent on suction plateau
        # If can not find suction peak, err = 0, DMp = 0.0
        Err = 0.0
        DMp = 0.0
        iL  = self.xf_dict['L'][1]
        if iL!=0:
            i1 = self.xf_dict['1'][1]
            xL  = self.xf_dict['L'][2]

            Cp0 = np.array([xL, self.getValue('L','Cp')])
            Cp1 = np.array([x1, self.getValue('1','Cp')])

            Mw0 = self.getValue('L','Mw')
            Mw1 = self.getValue('1','Mw')
            lL1 = x1-xL

            for i in np.arange(iL, i1, 1):

                vec = np.array([X[i], self.Cp[i]])
                s, _ = ratio_vec(Cp0, Cp1, vec)
                Err += s*(X[i+1]-X[i])

                tt = (X[i]-xL)/lL1
                ss = (1-tt)*Mw0 + tt*Mw1
                DMp = max(DMp, ss-self.Mw[i])

        self.xf_dict['Err'][1] = abs(Err)*rr
        self.xf_dict['DMp'][1] = DMp

        #* CLU => CL of upper surface
        CLU = 0.0
        for i in np.arange(self.iLE, len(X)-1, 1):
            CLU += 0.5*(self.Cp[i]+self.Cp[i+1])*(X[i+1]-X[i])

        self.xf_dict['CLU'][1] = abs(CLU)*rr

        #* kaf => average Mw slope of the aft upper surface (3/N~T)
        xN  = self.xf_dict['N'][2]
        mN  = self.getValue('N','Mw')
        xT  = self.xf_dict['T'][2]
        mT  = self.getValue('T','Mw')
        if xN < 0.1:
            xN = self.xf_dict['3'][2]
            mN = self.getValue('3','Mw')

        self.xf_dict['kaf'][1] = (mT-mN)/(xT-xN)

    def extract_features(self):
        '''
        Extract flow features list in the dictionart.
        '''
        self.locate_basic()
        self.locate_sep()
        self.locate_geo()
        i_1 = self.locate_shock()
        self.locate_BL(i_1)
        self.aux_features()

    #TODO: output features
    def output_features(self, fname="feature2d.txt", append=True, keys_=None):
        '''
        Output all features to file.

        ### Inputs:
        ```text
        keys:  list of key strings for output. None means default.
        ```

        ### Output order:
        ```text
        feature:    keys of feature dictionary
        key:        value or ('X', 'Cp', 'Mw', 'Tw', 'Hi', 'Hc', 'dudy')
        ```
        '''
        if keys_ is not None:
            keys = copy.deepcopy(keys_)
        else:
            keys = ['X', 'Mw', 'Hc']

        if not os.path.exists(fname):
            append = False

        if not append:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')

        for feature in self.xf_dict.keys():
            
            if len(self.xf_dict[feature])==2:
                value = self.getValue(feature)
                f.write('%10s   %15.6f \n'%(feature, value))
                continue

            for key in keys:
                name = key+'-'+feature
                value = self.getValue(feature, key)
                f.write('%10s   %15.6f \n'%(name, value))

        f.close()


class PhysicalXfoil(PhysicalSec):
    '''
    Extract features from Xfoil (low speed) results.
    '''
    def __init__(self, Minf, AoA, Re):
        super().__init__(Minf, AoA, Re)

    def setdata(self, x, y, Cp):
        '''
        Set the data of this foil or section.

        x,y,Cp: list, start from upper surface trailing edge (order from xfoil)
        '''
        x_  = list(reversed(x))
        y_  = list(reversed(y))
        Cp_ = list(reversed(Cp))

        n = int(len(x_)/2)
        x_  = x_[:n] + [0.0] + x_[n:]
        y_  = y_[:n] + [0.5*(y_[n]+y_[n-1])] + y_[n:]
        Cp_ = Cp_[:n] + [0.5*(Cp_[n]+Cp_[n-1])] + Cp_[n:]

        self.x = np.array(x_)
        self.y = np.array(y_)
        self.Cp = np.array(Cp_)
        self.Mw = self.Cp2Mw()

        iLE = np.argmin(self.x)
        
        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')
        gu  = interp1d(self.x[iLE:], self.y [iLE:], kind='cubic')
        x_  = np.append(self.x[iLE:0:-1], self.x[0])
        y_  = np.append(self.y[iLE:0:-1], self.y[0])
        gl  = interp1d(x_, y_, kind='cubic')

        self.xx = np.arange(0.0, 1.0, 0.001)
        self.yu = gu(self.xx)
        self.yl = gl(self.xx)
        self.mu = fmw(self.xx)

        self.iLE = iLE

    def extract_features(self):
        '''
        Extract flow features list in the dictionart.
        '''
        self.locate_basic()
        self.locate_geo()

    def output_features(self, fname="feature-xfoil.txt", append=True):
        '''
        Output all features to file.

        ### Output order:
        ```text
        feature:    keys of feature dictionary
        key:        'X', 'Mw', 'Cp'
        ```
        '''
        keys = ['X','Mw','Cp']
        features = ['L', 'T', 'Q', 'M', 'Cu', 'Cl', 'tu', 'tl', 'tm']

        if not os.path.exists(fname):
            append = False

        if not append:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')

        for feature in features:
            
            if len(self.xf_dict[feature])==2:
                value = self.getValue(feature)
                f.write('%10s   %15.6f \n'%(feature, value))
                continue

            for key in keys:
                name = key+'-'+feature
                value = self.getValue(feature, key)
                f.write('%10s   %15.6f \n'%(name, value))

        f.close()


class PhysicalTSFoil(PhysicalSec):
    '''
    Extract features from pyTSFoil (transonic speed) results.
    '''
    def __init__(self, Minf, AoA, Re):
        super().__init__(Minf, AoA, Re)

    def setdata(self, xu, yu, xl, yl, cpu, cpl, mwu, mwl):
        '''
        Set the data of this foil or section.

        ### Note:
        ```text
        xu, yu, xl, yl, cpu, cpl:   ndarray from pyTSFoil
        mwu, mwl:   ndarray from pyTSFoil (do not need built-in Cp2Mw)
        ```
        '''
        cp1u = cpu[-1] + (1-xu[-2])/(xu[-1]-xu[-2])*(cpu[-1]-cpu[-2])
        cp1l = cpl[-1] + (1-xl[-2])/(xl[-1]-xl[-2])*(cpl[-1]-cpl[-2])
        cp1 = 0.5*(cp1u+cp1l)

        self.x  = np.array([1.0]+list(reversed(list(xl[1:]))) + list(xu)+[1.0])
        self.y  = np.array([0.0]+list(reversed(list(yl[1:]))) + list(yu)+[0.0])
        self.Cp = np.array([cp1]+list(reversed(list(cpl[1:]))) + list(cpu)+[cp1])
        
        '''
        mw1u = mwu[-1] + (1-xu[-2])/(xu[-1]-xu[-2])*(mwu[-1]-mwu[-2])
        mw1l = mwl[-1] + (1-xl[-2])/(xl[-1]-xl[-2])*(mwl[-1]-mwl[-2])
        mw1 = 0.5*(mw1u+mw1l)
        self.Mw = np.array([mw1]+list(reversed(list(mwl[1:]))) + list(mwu)+[mw1])
        '''
        self.Mw = self.Cp2Mw()

        iLE = np.argmin(self.x)

        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')
        gu  = interp1d(self.x[iLE:], self.y [iLE:], kind='cubic')
        x_  = np.append(self.x[iLE:0:-1], self.x[0])
        y_  = np.append(self.y[iLE:0:-1], self.y[0])
        gl  = interp1d(x_, y_, kind='cubic')

        self.xx = np.arange(0.0, 1.0, 0.001)
        self.yu = gu(self.xx)
        self.yl = gl(self.xx)
        self.mu = fmw(self.xx)

        self.iLE = iLE

    def extract_features(self):
        '''
        Extract flow features list in the dictionart.
        '''
        self.locate_basic()
        self.locate_geo()
        self.locate_shock()
        self.aux_features()

    def output_features(self, fname="feature2d.txt", append=True):
        '''
        Output all features to file.

        ### Output order:
        ```text
        feature:    keys of feature dictionary
        key:        'X', 'Mw', 'Cp'
        ```
        '''

        keys = ['X','Mw','Cp']
        features = ['L', 'T', 'Q', 'M', 'F', '1', 'U', '3',
                    'Cu', 'Cl', 'tu', 'tl', 'tm', 'L13',
                    'lSW', 'DCp', 'Err', 'CLU', 'kaf']

        if not os.path.exists(fname):
            append = False

        if not append:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')

        for feature in self.xf_dict.keys():
            
            if len(self.xf_dict[feature])==2:
                value = self.getValue(feature)
                f.write('%10s   %15.6f \n'%(feature, value))
                continue

            for key in keys:
                name = key+'-'+feature
                value = self.getValue(feature, key)
                f.write('%10s   %15.6f \n'%(name, value))

        f.close()


#* ========================================
#* Supportive functions
#* ========================================

def ratio_vec(x0, x1, x):
    '''
    Calculate distance s to vector x1-x0.

    ### Inputs:
    ```text
    x0, x1: ndarray, start and end point of the vector
    x:      ndarray, current point
    ```

    ### Return:
    ```text
    s:  distance to line
    t:  ratio of (projected |x0x|) / |x0x1|
    ```
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

    ### Inputs:
    ```text
    x, y: points of curve (list or ndarray)
    ```

    ### Return:
    ```text
    curv: ndarray
    ```
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
