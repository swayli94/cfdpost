# cfdpost

CFD post procedures.

## # post

    Frequently used CFD post procedures.

### [ post_foil_cfl3d ]

    post_foil_cfl3d(path, j0, j1, nHi=40, fname="feature2d.txt")

    Read in CFL3D foil result and extract flow features.
        path:   folder that contains the output files
        j0:     j index of the lower surface TE
        j1:     j index of the upper surface TE
        nHi:    maximum number of mesh points in k direction for boundary layer

    Single C-block cfl3d.prt index
        i : 1 - 1   symmetry plane
        j : 1 - nj  far field of lower surface TE to far field of upper surface TE
        k : 1 - nk  surface to far field


## # cfdresult

    Read in CFD results.

### cfl3d

    Read in CFL3D results.

## # feature2d

    Extract 2D flow features.

### FeatureSec

    Extract flow features of airfoils or wing sections.
