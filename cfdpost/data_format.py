'''
This file defines the data format of ONE airfoil.

------------------------
ASCII File name:    id.dat (id is an integer)
Number format:      %20d, %20.12E, %50s

Line    Content                             Description
1       ID YYYY-MM-DD HH:MM:SS              ID & creation time
2       INFORMATION                         information of this file (string, len=200)
3       N-ATTRIBUTE                         number of attributes in the first part
4       N-ZONE                              number of zones
5       NI NJ NK                            dimensions of each zone
6       N-VARIABLE                          number of variables in each zone
7       NAME-VARIABLES                      names of variables (each name is a string, len=50)
8       # BLANK
9       # BLANK
10      # BLANK
10+d    NAME    FLOAT-ATTRIBUTE             name (string, len=50) and float value of attributes, d=1,...,N-ATTRIBUTE

D0+1    NAME-ZONE                           string, len=200, 'zone [NAME]', D0=10+N-ATTRIBUTE
D0+2                                        for k in range(NK):
                                                for j in range(NJ):
                                                    for i in range(NI):
D0+d    FLOAT-VARIABLES                                 float-1, float-2, ..., float-N-VARIABLE


------------------------
Binary File name:   id.bin (id is an integer)

Index       Content                     Format
1           ID                          i
2           YYYY-MM-DD HH:MM:SS         20s
3           N-ATTRIBUTE                 i
4           N-ZONE                      i
5           NI                          i
6           NJ                          i
7           NK                          i
8           N-VARIABLE                  i
9           INFO                        200s
N0+i        ATTRIBUTE[i]                d       N0 = 9
N1+i        NAME-ATTRIBUTE[i]           50s     N1 = N0 + N-ATTRIBUTE
N2+i        NAME-ZONE[i]                200s    N2 = N1 + N-ATTRIBUTE
N3+i        NAME-VARIABLE[i]            50s     N3 = N2 + N-ZONE
N4+n        VARIABLE[z,k,j,i,v]         d       N4 = N3 + N-VARIABLE, z,k,j,i,v loop
                                        
------------------------
Format      C-type      Fortran-type        Python-type     Bytes
i           int         integer             integer         4
d           double      real*8              float           8
s           char[]      character           string          1
4s          char[4]     character(len=4)    %4s             4
'''
import os
import struct as st
import time

import numpy as np


def save_ascii(ID: int, ATTRIBUTES: np.array, ZONES: np.array, 
                NAME_ATTRS: list, NAME_VARS: list, NAME_ZONE=[],
                INFO='', PATH='.', PREFIX='', SUFFIX='', forTecplot=False):
    '''
    Save data to file [PATH\{PREFIX}ID{SUFFIX}.dat]

    ### Inputs:
    ```text
    ID:             integer
    ATTRIBUTES:     ndarray [N-ATTRIBUTE]
    ZONES:          ndarray [N-ZONE,NK,NJ,NI,N-VARIABLE]
    NAME_ATTRS:     list, len = N-ATTRIBUTE, string format %50s
    NAME_VARS:      list, len = N-VARIABLE, string format %50s
    NAME_ZONE:      list, len = N-ZONE, string format
    INFO:           string
    PATH:           file directory
    forTecplot:     if True, change format for Tecplot
    ```
    '''

    n_attr = len(NAME_ATTRS)
    n_zone = ZONES.shape[0]
    nk     = ZONES.shape[1]
    nj     = ZONES.shape[2]
    ni     = ZONES.shape[3]
    n_vars = ZONES.shape[4]

    if forTecplot:
        filename = '%s%d%s-tecplot.dat'%(PREFIX, ID, SUFFIX)
        header   = '# '
    else:
        filename = '%s%d%s.dat'%(PREFIX, ID, SUFFIX)
        header   = ''

    f = open(os.path.join(PATH, filename), 'w')

    f.write(header+'%20d  '%(ID))
    f.write(header+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    f.write('\n')

    f.write(header+'  %s\n'%(INFO.strip()))
    f.write(header+'%20d\n'%(n_attr))
    f.write(header+'%20d\n'%(n_zone))
    f.write(header+' %19d %19d %19d\n'%(ni,nj,nk))
    f.write(header+'%20d\n'%(n_vars))

    if forTecplot:
        f.write('Variables=')
        for name in NAME_VARS:
            f.write(' "%s"'%(name.strip()[:50]))
        f.write('\n')
    else:
        for name in NAME_VARS:
            f.write('  %s'%(name.strip()[:50]))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('\n')

    for i in range(n_attr):
        f.write(header+'%20s %19.12E\n'%(NAME_ATTRS[i].strip()[:50], ATTRIBUTES[i]))

    for z in range(n_zone):

        if forTecplot:
            if len(NAME_ZONE)==n_zone:
                f.write('zone T= "%s"\n'%(NAME_ZONE[z].strip()))
            else:
                f.write('zone T= "%d"\n'%(z+1))
        else:
            if len(NAME_ZONE)==n_zone:
                f.write('zone %s\n'%(NAME_ZONE[z].strip()))
            else:
                f.write('zone %d\n'%(z+1))

        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    for v in range(n_vars):
                        f.write(' %19.12E'%(ZONES[z,k,j,i,v]))
                    f.write('\n')

    f.close()

def load_ascii(filename: str, forTecplot=False):
    '''
    ### Return:
    ```text
    ID:             ID in file
    DATE:           string, creation date
    TIME:           string, creation time
    INFO:           string
    NAME_VARS:      list, len = N-VARIABLE
    NAME_ATTRS:     list, len = N-ATTRIBUTE
    ATTRIBUTES:     ndarray [N-ATTRIBUTE]
    NAME_ZONE:      list, len = N-ZONE
    ZONES:          ndarray [N-ZONE,NK,NJ,NI,N-VARIABLE]
    ```
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        nLINE = len(lines)

        if forTecplot:
            n0 = 7
            i0 = 1
        else:
            n0 = 10
            i0 = 0

        line = lines[0].split()
        ID   = int(line[i0+0])
        DATE = line[i0+1]
        TIME = line[i0+2]
        INFO = lines[1].strip()

        n_attr = int(lines[2].split()[-1])
        n_zone = int(lines[3].split()[-1])
        line   = lines[4].split()
        ni     = int(line[i0+0])
        nj     = int(line[i0+1])
        nk     = int(line[i0+2])
        n_vars = int(lines[5].split()[-1])

        NAME_VARS = lines[6].split()[i0:]

        NAME_ATTRS = []
        ATTRIBUTES = np.zeros(n_attr)
        for i in range(n_attr):
            line = lines[n0+i].split()
            NAME_ATTRS.append(line[i0+0])
            ATTRIBUTES[i] = float(line[i0+1])

        NAME_ZONE = []
        ZONES = np.zeros([n_zone, nk, nj, ni, n_vars])
        iLINE = n0+n_attr

        for z in range(n_zone):

            if forTecplot:
                line = lines[iLINE].split('"')
                NAME_ZONE.append(line[-1])
            else:
                NAME_ZONE.append(lines[iLINE][5:].strip())

            iLINE += 1

            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        line = lines[iLINE].split()
                        iLINE += 1
                        for v in range(n_vars):
                            ZONES[z,k,j,i,v] = float(line[v])
  
        return ID, DATE, TIME, INFO, NAME_VARS, NAME_ATTRS, ATTRIBUTES, NAME_ZONE, ZONES

def save_binary(ID: int, ATTRIBUTES: np.array, ZONES: np.array, 
                NAME_ATTRS: list, NAME_VARS: list, NAME_ZONE=[],
                INFO='', PATH='.', PREFIX='', SUFFIX=''):
    '''
    Save data to file [PATH\{PREFIX}ID{SUFFIX}.bin]

    ### Inputs:
    ```text
    ID:             integer
    ATTRIBUTES:     ndarray [N-ATTRIBUTE]
    ZONES:          ndarray [N-ZONE,NK,NJ,NI,N-VARIABLE]
    NAME_ATTRS:     list, len = N-ATTRIBUTE, string format %50s
    NAME_VARS:      list, len = N-VARIABLE, string format %50s
    NAME_ZONE:      list, len = N-ZONE, string format %200s
    INFO:           string, %200s
    PATH:           file directory
    ```
    '''

    n_attr = len(NAME_ATTRS)
    n_zone = ZONES.shape[0]
    nk     = ZONES.shape[1]
    nj     = ZONES.shape[2]
    ni     = ZONES.shape[3]
    n_vars = ZONES.shape[4]

    TIME = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())

    filename = '%s%d%s.bin'%(PREFIX, ID, SUFFIX)

    f = open(os.path.join(PATH, filename), 'wb')
    f.write(st.pack('i',    ID))
    f.write(st.pack('20s',  TIME.encode('ascii')))
    f.write(st.pack('i',    n_attr))
    f.write(st.pack('i',    n_zone))
    f.write(st.pack('i',    ni))
    f.write(st.pack('i',    nj))
    f.write(st.pack('i',    nk))
    f.write(st.pack('i',    n_vars))
    f.write(st.pack('200s', INFO.encode('ascii')))

    for i in range(n_attr):
        f.write(st.pack('d',   ATTRIBUTES[i]))

    for i in range(n_attr):
        f.write(st.pack('50s', NAME_ATTRS[i].encode('ascii')))

    if len(NAME_ZONE)==n_zone:
        for z in range(n_zone):
            if len(NAME_ZONE[z])>=200:
                print('Warning [save_binary]: length of zone name should < 200, now = %d'%(len(NAME_ZONE[z])))
            f.write(st.pack('200s', NAME_ZONE[z].encode('ascii')))
    else:
        for z in range(n_zone):
            name = 'ZONE-%d\n'%(z+1)
            f.write(st.pack('200s', name.encode('ascii')))

    for i in range(n_vars):
        f.write(st.pack('50s', NAME_VARS[i].encode('ascii')))

    for z in range(n_zone):
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    for v in range(n_vars):
                        f.write(st.pack('d', ZONES[z,k,j,i,v]))

    f.close()

def load_binary(filename: str):
    '''
    ### Return:
    ```text
    ID:             ID in file
    DATE:           string, creation date
    TIME:           string, creation time
    INFO:           string
    NAME_VARS:      list, len = N-VARIABLE
    NAME_ATTRS:     list, len = N-ATTRIBUTE
    ATTRIBUTES:     ndarray [N-ATTRIBUTE]
    NAME_ZONE:      list, len = N-ZONE
    ZONES:          ndarray [N-ZONE,NK,NJ,NI,N-VARIABLE]
    ```
    '''
    with open(filename, 'rb') as f:

        x00     = b'\x00'.decode()

        ID,     = st.unpack('i', f.read(4))
        DATE,   = st.unpack('10s', f.read(10))
        DATE    = DATE.decode()
        TIME,   = st.unpack('10s', f.read(10))
        TIME    = TIME.decode().strip()
        n_attr, = st.unpack('i', f.read(4))
        n_zone, = st.unpack('i', f.read(4))
        ni,     = st.unpack('i', f.read(4))
        nj,     = st.unpack('i', f.read(4))
        nk,     = st.unpack('i', f.read(4))
        n_vars, = st.unpack('i', f.read(4))
        INFO,   = st.unpack('200s', f.read(200))
        INFO    = INFO.decode().strip(x00)
        INFO    = INFO.strip()

        ATTRIBUTES = np.zeros(n_attr)
        for i in range(n_attr):
            ATTRIBUTES[i], = st.unpack('d', f.read(8))

        NAME_ATTRS = []
        for i in range(n_attr):
            name, = st.unpack('50s', f.read(50))
            NAME_ATTRS.append(name.decode().strip(x00))

        NAME_ZONE = []
        for z in range(n_zone):
            name, = st.unpack('200s', f.read(200))
            NAME_ZONE.append(name.decode().strip(x00))

        NAME_VARS = []
        for i in range(n_vars):
            name, = st.unpack('50s', f.read(50))
            NAME_VARS.append(name.decode().strip(x00))

        ZONES = np.zeros([n_zone,nk,nj,ni,n_vars])
        for z in range(n_zone):
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        for v in range(n_vars):
                            ZONES[z,k,j,i,v], = st.unpack('d', f.read(8))

        return ID, DATE, TIME, INFO, NAME_VARS, NAME_ATTRS, ATTRIBUTES, NAME_ZONE, ZONES


if __name__ == '__main__':

    print('Test data_format.py')
    print()

    n_attr = 3
    n_zone = 20
    n_vars = 5
    ni = 400
    nj = 100
    nk = 1
    ID = 1

    ATTRIBUTES = np.linspace(1,n_attr,num=n_attr)
    ZONES = np.random.rand(n_zone, nk, nj, ni, n_vars)
    NAME_ATTRS = ['attr-%d'%(i) for i in range(n_attr)]
    NAME_VARS  = ['var-%d'%(i)  for i in range(n_vars)]
    NAME_ZONE  = ['TEST ZONE #%d'%(i) for i in range(n_zone)]
    INFO = 'Testing'


    t1 = time.process_time()
    save_ascii(ID, ATTRIBUTES, ZONES, NAME_ATTRS, NAME_VARS, NAME_ZONE=NAME_ZONE, INFO=INFO, PATH='.')
    t2 = time.process_time()

    ID, DATE, TIME, INFO, NAME_VARS, NAME_ATTRS, ATTRIBUTES, NAME_ZONE, ZONES = load_ascii('1.dat')
    t3 = time.process_time()

    print(t2-t1, t3-t2)
    print(ID, DATE, TIME, INFO)
    print(NAME_VARS)
    print(NAME_ATTRS)
    print(ATTRIBUTES)
    print()
    
    
    save_binary(ID, ATTRIBUTES, ZONES, NAME_ATTRS, NAME_VARS, NAME_ZONE=NAME_ZONE, INFO=INFO, PATH='.')
    t4 = time.process_time()

    ID, DATE, TIME, INFO, NAME_VARS, NAME_ATTRS, ATTRIBUTES, NAME_ZONE, ZONES = load_binary('1.bin')
    t5 = time.process_time()

    print(t4-t3, t5-t4)
    print(ID, DATE, TIME, INFO)
    print(NAME_VARS)
    print(NAME_ATTRS)
    print(ATTRIBUTES)
    print()


