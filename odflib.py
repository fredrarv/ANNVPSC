import os
import tempfile
#import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import pylab as plt
import auswert

#__all__ = ['Orientations', 'ODF', 'read_orifile', 'read_oimfile',
#           'ori_from_oimfile', 'mdf_from_oimfile']



def asnumber(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def read_orifile(fname):
    """Reads a .ori file and return an Orientations object."""
    f = open(fname, 'r') if isinstance(fname, str) else fname
    label = f.readline().strip()
    PHI2, lmax = f.readline().split()
    lmax = int(lmax)
    if PHI2 != 'PHI2':
        raise IOError('%s: line 2: must start with PHI2, got "%s"' %
                      (f.name, PHI2))
    n, w0, binsize = f.readline().split()
    n, w0, binsize = int(n), float(w0), float(binsize)
    a = []
    for i in range(n):
        a.append([float(v) for v in f.readline().split()])
    a = np.array(a)
    if a.shape[1] == 3:
        return Orientations(a[:,:3], label=label, lmax=lmax,
                            weight=w0, binsize=binsize)
    elif a.shape[1] == 4 or np.allclose(a[:,4], 0.0):
        return Orientations(a[:,:3], label=label, lmax=lmax,
                            weight=a[:,3], binsize=binsize)
    else:
        return Orientations(a[:,:3], label=label, lmax=lmax,
                            weight=a[:,3], binsize=a[:,4])



def copy_datafiles(files=('libbu.dat', 'liba23.dat', 'liba1.dat',
                          'libmln.dat', 'libxk.dat')):
    """Copy datafiles needed by auswert to current directory, removing
    them automatically at program exit."""
    def delfile(filename):
        os.remove(filename)
    for filename in files:
        if not os.path.exists(filename):
            import shutil
            import atexit
            src = os.path.join(os.path.dirname(__file__), 'data', filename)
            shutil.copyfile(src, filename)
            atexit.register(delfile, filename)



def odffig(odf, boundaries=None, colors=None, col_over='k', col_under='w',
           col_contour='k'):
    """Create and return a figure of given ODF.

    Arguments
    ---------
    odf: float 3d array
        The ODF, normally with shape 19x19x19.
    boundaries: None | sequence of floats
        Sequence containing the contour levels.  Must be strictly
        increasing.  If None, the default boundaries [1., 2., 4., 7.,
        10.] will be used.
    colors: None | sequence of colors
        Sequence of colors. Must be of the same length as *boundaries*.
        If None, a grayscale will be used.
    col_over: color
        Color to be used for values above largest value in *boundaries*.
        Default is black.
    col_under: color
        Color to be used for values below smallest value in *boundaries*.
        Default is white.
    col_contour: None | color | sequence of colors.
        Color of contour lines.  If None, no contour lines will be drawn.
        If a single color, that color will be used for all contour lines.
        Otherwise it must be a sequence of same length as *boundaries*
        specifying the color or the individual contour lines.
    """
    if boundaries is None:
        boundaries = [1., 2., 4., 7., 10.]
    if colors is None:
        N = len(boundaries)
        colors = ['%.3f' % v for v in 
                  1.0 - np.arange(N * 1.0)/(N + 1.0) - 1.0/(N + 1.0)]

    cmap = mpl.colors.ListedColormap(colors)
    cmap.set_over(color=col_over)
    cmap.set_under(color=col_under)
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    
    hi, lo = odf.max(), odf.min()
    if hi > boundaries[-1]:
        boundaries2 = [lo] + list(boundaries) + [hi]        
    else:
        boundaries2 = [lo] + list(boundaries) + [boundaries[-1]*1.1]

    colors2 = [col_under] + list(colors) + [col_over]    
    cmap2 = mpl.colors.ListedColormap(colors2)
    norm2 = mpl.colors.BoundaryNorm(boundaries2, cmap2.N)

    nphi2, nPhi, nphi1 = odf.shape
    cols = np.floor(np.sqrt(nphi1))
    rows = np.ceil(nPhi/cols)
    indmax = flat2ind(odf.shape, odf.argmax())
    x0 = 0.5/(cols + 2.0) - 0.1/6.           # 0.4/6.
    x1 = 0.5/(cols + 2.0)                    # 0.5/6.
    x2 = 0.5 - 0.5/6.                        # 2.5/6.
    x3 = (cols + 0.5)/(cols + 2.0) + 0.3/6.  # 4.8/6.
    x4 = (cols + 0.5)/(cols + 2.0) + 0.45/6. # 4.95/6.
    y0 = (rows + 0.5)/(rows + 1.0) + 0.1/6.  # 5.6/6.
    y1 = (rows + 0.5)/(rows + 1.0)           # 5.5/6.

    fig = plt.figure(figsize=(8, 8), facecolor='w')
    fig.text(x2, y0, r'$\varphi_2 \/ \mathrm{constant}$', ha='center', size='large')
    fig.text(x1, y0, r'$\varphi_1\longrightarrow$', size='large')
    fig.text(x0, y1, r'$\Phi$', ha='center', va='top', size='large')
    fig.text(x0 - 0.04/6., y1 - 0.1/6., r'$\longleftarrow$', rotation='vertical', va='top', size='large')
    fig.text(x3, 5.15/6., r'Levels:')
    fig.text(x3, 2.5/6., r'Fmax = %.1f' % hi)
    fig.text(x3, 2.3/6., r'at')
    fig.text(x4, 2.3/6., r'$\varphi_1 = %d^\circ$' % (indmax[0]*5, ))
    fig.text(x4, 2.1/6., r'$\Phi = %d^\circ$' % (indmax[1]*5, ))
    fig.text(x4, 1.9/6., r'$\varphi_2 = %d^\circ$' % (indmax[2]*5, ))

    n = 0
    for i in np.arange(rows):
        for j in np.arange(cols):
            if n >= nphi2:
                break
            left = (j + 0.5)/(cols + 2.)
            bottom = (4.5 - i)/(rows + 1.)
            width = 1./(cols + 2.)
            height = 1./(rows + 1.)
            ax = fig.add_axes([left, bottom, width, height], 
                              xticks=[], yticks=[], aspect='equal')
            ax.contourf(odf[n], cmap=cmap2, norm=norm2, 
                        levels=boundaries2, origin='upper')
            if col_contour is not None:
                ax.contour(odf[n], colors=col_contour, levels=boundaries, 
                           origin='upper')
            n += 1

    cbax = fig.add_axes([(cols + 1.)/(cols + 2.), 0.5,
                         0.15/(cols + 2.0), 2./6.])
    cb2 = mpl.colorbar.ColorbarBase(cbax, 
                                    cmap=cmap,
                                    norm=norm,
                                    format='%.1f',
                                    # to use 'extend', you must
                                    # specify two extra boundaries:
                                    boundaries=boundaries2,
                                    extend='both',
                                    ticks=boundaries, # optional
                                    spacing='proportional',
                                    orientation='vertical',
                                    drawedges=True,
                                    )

    return fig
            


def readodf(f):
    """Read odf from file object 'f' and return it as a 19x19x19 array."""
    odf = np.empty((19, 19, 19))
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    for i in range(19):
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        for j in range(19):
            line = f.readline()
            odf[i,j,:] = [float(v) for v in line.split()[1:-1]]
    return odf


#def odfsample(odf):
#    """Sample from given odf."""
#    nx, ny, nz = odf.shape
#    s = odf.sum()


def flat2ind(shape, flat, order='C'):
    """Convert flatten scalar index *flat* to a multidimensional index
    for given shape."""
    ind = np.empty((len(shape), ), dtype=int)
    for i, n in enumerate(shape):
        j = i if order == 'C' else len(shape) - 1 - i
        ind[j] = flat % n
        flat //= n
    return ind
    
    
    
    

class Orientations(object):
    """Reads a .ori-file and returns a Orientations object.

    Arguments
    ---------
    angles: Nx3 array_like
        Euler angles in degree
    label: string
        A label for the data (up to 10 characters)
    lmax: int
        Order of expansion.
    weight: 1.0 | sequence of floats
        Weight of Euler angles.
    binsize: float | sequence of floats
        Bin size bin in degrees. If *binsize* is a float, all bins will have
        this size, otherwise *binsize* must be a array of same length as
        *angles*.
    """
    def __init__(self, angles, label='', lmax=22, weight=1.0, binsize=5.0):
        self.set_angles(angles, weight, binsize)
        self.set_label(label)
        self._lmax = int(lmax)

    def set_angles(self, angles, weight=1.0, binsize=5.0):
        """Sets Euler angles, weight and bin size."""
        a = np.array(angles)
        if a.ndim != 2:
            raise TypeError('*angles* must be a 2d array_like')
        if a.shape[1] != 3:
            raise TypeError('*angles* must have shape Nx3')
        try:
            w = float(weight)
        except TypeError:
            w = np.array(weight)
            if w.ndim != 1:
                raise TypeError('*weight* must be a float or 1d array_like')
            if len(w) != len(a):
                raise TypeError('*weight* must be of same length as *angles*')

        try:
            p = float(binsize)
        except TypeError:
            p = np.array(binsize)
            if p.ndim != 1:
                raise TypeError('*binsize* must be a float or 1d array_like')
            if len(p) != len(a):
                raise TypeError('*binsize* must be of same length as *angles*')
        self._angles = a
        self._weight = w
        self._binsize = p

    def set_label(self, label):
        """Sets the label."""
        lab = str(label)
        if '\n' in lab:
            raise ValueError('*label* cannot contain a newline')
        self._label = lab

    def __len__(self):
        return len(self._angles)

    def __str__(self):
        if self._label:
            label = self._label
        else:
            import time
            label = 'Generated by odflib, ' + time.asctime()
        if isinstance(self._weight, float):
            w0, w = self._weight, np.zeros(len(self._angles), )
        else:
            w0, w = 0.0, self._weight

        if isinstance(self._binsize, float):
            binsize, psi = self._binsize, np.zeros(len(self._angles), )
        else:
            binsize, psi = 0.0, self._binsize

        s = []
        s.append(label)
        s.append('PHI2 %d' % self._lmax)
        s.append('%d  %.2f  %.2f' % (len(self._angles), w0, binsize))
        for (phi1, Phi, phi2), w, p in zip(self._angles, w, psi):
            s.append('%8.2f %8.2f %8.2f    %8.3f    %8.3f' %
                     (phi1, Phi, phi2, w, p))

        return '\n'.join(s) + '\n'

    def write(self, fname):
        """Writes orientations to file *fname*."""
        if isinstance(fname, str):
            with open(fname, 'w') as f:
                f.write(str(self))
        else:
            fname.write(str(self))

    def write_ckofile(self, filename=None, dir='.'):
        """Writes C-coefficients to *filename* using the gauss_odf()
        routine by Olaf Engler.  If *filename* is None, a unique
        filename will be generated in directory *dir*.  The file name
        is returned."""
        ckofile = orifile = None
        try:
            if filename:
                ckofile = filename
            else:
                # if not given, create unique ckofile name
                ckofd, ckofile = tempfile.mkstemp(
                    prefix='odflib-', suffix='.cko', dir=dir)
                ckofile = os.path.relpath(ckofile)
                os.close(ckofd)

            # write orientations to temporary orifile
            fd, orifile = tempfile.mkstemp(
                prefix='odflib-', suffix='.ori', dir=dir)
            f = os.fdopen(fd, 'w')
            f.write(str(self))
            f.flush()

            copy_datafiles()
            if auswert.gauss_odf_wrap(orifile, ckofile):
                raise IOError('cannot open orientation file')
            f.close()

        except:
            # remove newly created ckofile on error
            if not filename and ckofile:
                os.remove(ckofile)
            raise
        finally:
            # clean up temporary orifile
            if orifile:
                os.remove(orifile)

        return ckofile


    angles = property(lambda self: self._angles,
                      doc='Euler angles (degrees).')

    label = property(lambda self: self._label,
                     set_label,
                     doc='Label.')

    lmax = property(lambda self: self._lmax,
                    lambda self, value: setattr(self, '_lmax', int(value)),
                    'Order of spherical harmonic expansion.')

    weight = property(lambda self: self._weight,
                      lambda self, value: self.set_angles(self._angles, value, self._binsize),
                      doc='Weight')

    binsize = property(lambda self: self._binsize,
                      lambda self, value: self.set_angles(self._angles, self._weight, value),
                      doc='Bin size (degrees)')


    def get_odf(self, ckofile=None, odffile=None, dir='.'):
        """Returns a 19x19x19 ODF filtered with the gauss_odf() and
        cko2odf() routines by Olaf Engler.  If file names are given
        for *ckofile* or *odffile*, the corresponding file will be
        stored with the given name, otherwise directory *dir* will be
        used for temporary files."""
        del_ckofile = not bool(ckofile)
        del_odffile = not bool(odffile)
        try:
            ckofile = self.write_ckofile(ckofile, dir)
            try:
                if not odffile:
                    fd, odffile = tempfile.mkstemp(
                        prefix='odflib-', suffix='.odf', dir=dir)
                    os.close(fd)
                    copy_datafiles()
                #odf = np.pi**2 / 24.0 * auswert.cko2odf(ckofile, odffile)
                odf = auswert.cko2odf(ckofile, odffile)
            finally:
                if del_odffile:
                    os.remove(odffile)
        finally:
            if del_ckofile:
                os.remove(ckofile)
        return ODF(odf)

# SUBROUTINE NOT USED
#
#    def get_unfiltered_odf(self, shape=(19, 19, 19), binsize=(5.0, 5.0, 5.0)):
#        """Returns an unfiltered ODF calculated directly from Euler
#        angles.  *shape* is the shape of the returned ODF instance and
#        *binsize* is the bin size in degrees."""
#        angles = self._angles % (90., 90., 90.)
#        weights = self._weight * np.sin(np.pi * angles[:,1] / 180.0)
#        bins = [s*np.arange(n) for n, s in zip(shape, binsize)]
#        odf, edges =np.histogramdd(angles, bins, weights=weights)
#        return ODF(odf)
#
#
#    def sample(self, n):
#        """Simple sampling of *n* angles using Halton series.  *n* should be
#        not be larger than the number of orientations."""
#        if n > len(self):
#            raise ValueError('*n* must be less than the number of orientations.'
#                             ' Try to sample from ODF instead.')
#        h = halton(n)
#        inds = (h * len(self)).astype(int)
#        return self._angles[inds,:]






class ODF(object):
    """Class representing an orientation distribution function.

    Arguments
    ---------
    odf: 3d array_like
        ODF data to copy.
    orientations: 1d sequence | Orientations instance
        Array of orientations to generate ODF from.
    orifile: string
        Name of orientations .ori file to read and generate ODF from.
    ckofile: string
        Name of C-coefficient .cko file to read and generate ODF from.
    binsize: float | sequence of floats
        Bin size bin in degrees. If *binsize* is a float, all bins will have
        this size, otherwise *binsize* must be a array of same length as
        *angles*.
    """
    def __init__(self, odf=None, orientations=None, orifile=None, ckofile=None,
                 binsize=5.0):
        if sum([odf is not None, orientations is not None,
                orifile is not None, ckofile is not None]) != 1:
            raise ValueError(
                'One and only one of the *odf*, *orientations*, *orifile* and '
                '*ckgfile* arguments can be given')

        if odf is not None:
            data = np.array(odf)
            if data.ndim != 3:
                raise TypeError('*odf* must be a 3d array')
            self._data = data

        if orifile is not None:
            orientations = read_orifile(orifile)

        del_ckofile = False
        if orientations is not None:
            if not hasattr(orientations, 'angles'):
                orientations = Orientations(orientations)
            binsize = orientations.binsize
            ckofile = orientations.write_ckofile()
            del_ckofile = True

        if ckofile is not None:
            try:
                f = tempfile.NamedTemporaryFile(prefix='odflib-',
                                                suffix='.odf', dir='.', delete=False)
                
                odffile = f.name
                f.close()
                copy_datafiles()
#                self._data = np.pi**2 / 24.0 * auswert.cko2odf(ckofile, odffile)
                self._data = auswert.cko2odf(ckofile, odffile)
            finally:
                os.remove(odffile)

        if del_ckofile:
            os.remove(ckofile)

        self._binsize = binsize
        assert hasattr(self, '_data')
        
        
    def show(self, **kwargs):
        """Display ODF plot.  Keyword arguments are passed to odffig.
        A figure instance is returned."""
        o = np.zeros((19,19,19))
        for i in range(19):
            for j in range(19):
                for k in range(19):
                    o[i,j,k] = self._data[k,j,i]
        fig = odffig(o, **kwargs)
        #fig.show()
        #return fig

    def save(self, filename, **kwargs):
        """Write ODF plot to file.  Keyword arguments are passed to odffig.
        The figure instance is returned."""
        o = np.zeros((19,19,19))
        for i in range(19):
            for j in range(19):
                for k in range(19):
                    o[i,j,k] = self._data[k,j,i]
        fig = odffig(o, **kwargs)
        fig.savefig(filename)
        return fig


    data = property(lambda self: self._data,
                    doc='Direct access to ODF.')

# SUBROUTINES NOT USED
#
#    def sample(self, n=1, method='halton', sample_size=0):
#        """Returns a n x 3 array containing *n* samples from odf (in degree).
#
#        Arguments
#        ---------
#        n: int
#            Number of samples to return.
#        method: 'halton' | 'random'
#            Sampling method, must be either 'halton' (quasi-random) or
#            'random'.
#        sample_size: int
#            The approximate size of the internal cache for accelerated
#            sampling.  The default uses about 13.4 MB for an ODF of
#            size 19*19*19. (Currently not used).
#        """
#        odf = self._data
#        #sample = odflib_swig.ODFSample(odf, self._binsize, sample_size)
#
#        if method == 'halton':
#            #samp = sample.sample_halton(n)
#            samp = halton(n, 3)
#        elif method == 'random':
#            #samp = sample.sample(n)
#            samp = np.random.rand(n, 3)
#        else:
#            #del sample
#            raise ValueError('*method* must be "halton" or "random"')
#        #del sample
#
#        return self._sample(samp)
#
#
#    def get_data(self, unique=True):
#        """Returns a view of ODF data.  If *unique* is true, a view of
#        the periodic part of ODF data is returned."""
#        if unique:
#            if np.allclose((np.ones(3) * self._binsize) * self._data.shape,
#                           (90., 90., 90.)):
#                return self.data
#            elif np.allclose((np.ones(3) * self._binsize) *
#                             (self._data.shape - np.ones(3)),
#                             (90., 90., 90.)):
#                n0, n1, n2 = self._data.shape
#                return self.data[:n0-1,:n1-1,:n2-1]
#            else:
#                raise NotImplementedError
#        else:
#            return self.data
#
#    def get_dV(self, unique=True):
#        """Returns the volume of each bin.  If *unique* is true, the
#        volume of only the cells in the periodic unit are returned."""
#        n0, n1, n2 = self.get_data(unique).shape
#        d0, d1, d2 = np.ones((3, )) * self._binsize * np.pi / 180.0
#        Phi = np.arange(n1 + 1.0) * d1
#
#        dphi1 = (np.ones(n0) * d0)[np.newaxis, np.newaxis, :]
#        dPhi = np.diff(-np.cos(Phi))[np.newaxis, :, np.newaxis]
#        dphi2 = (np.ones(n2) * d2)[:, np.newaxis, np.newaxis]
#        dV = 32.0 / (8.0 * np.pi**2) * dphi1 * dPhi * dphi2
#        return dV
#
#
#    def get_Euler(self, unique=True):
#        """Returns the average Euler angles in each bin (in radians).
#        If *unique* is true, the Euler angles in the periodic unit are
#        returned"""
#        n0, n1, n2 = self.get_data(unique).shape
#        d0, d1, d2 = np.ones((3, )) * self._binsize * np.pi / 180.0
#        Phib = np.arange(n1 + 1.0) * d1
#
#        phi1 = ((np.arange(n0) + 0.5) * d0)[np.newaxis, np.newaxis, :]
#        Phi = ((np.diff(np.sin(Phib)) - np.diff(Phib * np.cos(Phib))) /
#               -np.diff(np.cos(Phib)))[np.newaxis, :, np.newaxis]
#        phi2 = ((np.arange(n2) + 0.5) * d2)[:, np.newaxis, np.newaxis]
#
#        return phi1, Phi, phi2
#
#
#    def _sample(self, R):
#        """Returns a n x 3 array containing *n* samples from odf (in degree).
#        """
#        #
#        # Formal set of equations used for sampling:
#        #   g = g(phi2, Phi, phi1)
#        #   G(phi2, Phi) = int_0^2*pi g * dphi1
#        #   G2(phi2) = int_0^pi G(phi2, Phi) * sin(Phi)*dPhi
#        #   F2(phi2) = int_0^phi2 G2(phi2') * dphi2'
#        #   FPhi(Phi|phi2) = int_0^Phi G(phi2, Phi) * sin(Phi')*dPhi'
#        #   F1(phi1|phi2,Phi) = int_0^phi1 g(phi2, Phi, phi1') * dphi1'
#        #
#
#        # split R into domain and fraction
#        R = np.array(R, copy=False, ndmin=2)
#        Ri = ((R * [4., 2., 4.]) // 1.0).astype(int) % [4, 2, 4]
#        Rf = (R * [4., 2., 4.]) % 1.0
#
#        data = self.get_data(unique=True)
#        n0, n1, n2 = data.shape
#        d0, d1, d2 = self._binsize * np.ones((3, )) * np.pi / 180.0
#        e0, e1, e2 = self.get_Euler(unique=True)
#        b0 = np.arange(n0 + 1.0) * d0
#        b1 = np.arange(n1 + 1.0) * d1
#        b2 = np.arange(n2 + 1.0) * d2
#        dV = self.get_dV()
#
#        # Remove negative values and normalise ODF
#        data = data.copy()
#        data[data < 0.0] = 0.0
#        dphi1 = (np.ones(n0) * d0)[np.newaxis, np.newaxis, :]
#        dPhi = (np.sin(e1) * d1)
#        dphi2 = (np.ones(n2) * d2)[:, np.newaxis, np.newaxis]
#        norm = (data * self.get_dV(unique=True)).sum()
#        data /= norm
#
#        g = data * dV
#        G = g.sum(axis=2)            # integrate over phi1
#        G2 = G.sum(axis=1)         # integrate over phi1 and Phi
#
#        # Sample phi2 from F2(phi2)
#        F2 = G2.cumsum()
#        assert abs(F2[-1] - 1.0) < 1e-5, 'Normalisation error'
#        F2 /= F2[-1]   # renormalise rounding errors
#        f2 = np.interp(Rf[:, 2], np.concatenate(([0.0], F2)), b2)
#        i2 = (f2 // d2).astype(int)
#
#        # Sample Phi from FPhi(Phi|phi2)
#        fPhi = []
#        for j2, R1 in zip(i2, Rf[:,1]):
#            FPhi = G[j2,:].cumsum() / G2[j2]
#            assert abs(FPhi[-1] - 1.0) < 1e-5, 'Normalisation error'
#            FPhi /= FPhi[-1]   # renormalise rounding errors
#            fPhi.append(np.interp(R1, np.concatenate(([0.0], FPhi)), b1))
#        fPhi = np.array(fPhi)
#        iPhi = (fPhi // d1).astype(int)
#
#        # Sample phi1 from F1(phi1|phi2,Phi)
#        f1 = []
#        for j2, jPhi, R0 in zip(i2, iPhi, Rf[:,0]):
#            F1 = g[j2,jPhi,:].cumsum() / G[j2, jPhi]
#            assert abs(F1[-1] - 1.0) < 1e-5, 'Normalisation error'
#            F1 /= F1[-1]   # renormalise rounding errors
#            f1.append(np.interp(R0, np.concatenate(([0.0], F1)), b0))
#        f1 = np.array(f1)
#        i1 = (f1 // d0).astype(int)
#
#        # ------------------------------------------------------------
#        #   About reduced Euler angles - from Olaf Engler
#        #
#        #   0 deg < phi1 <= 90 deg
#        #       phiR1 = phi1
#        #       PhiR  = Phi
#        #       phiR2 = phi2
#        #
#        #   90 deg < phi1 <= 180 deg
#        #       if ((phi1.GT.90.) .AND. (phi1.LE.180.)) then
#        #         phiR1 = 180.-phi1
#        #         phiR2 = 90.-phi2
#        #       endif
#        #
#        #   180 deg < phi1 <= 270 deg
#        #       if ((phi1.GT.180.) .AND. (phi1.LE.270.))  phiR1 = phi1-180.
#        #
#        #   270 deg < phi1 <= 360 deg
#        #       if (phi1.GT.270.) then
#        #         phiR1 = 360.-phi1
#        #         phiR2 = 90.-phi2
#        #       endif
#        # ------------------------------------------------------------
#        euler = np.array([f1, fPhi, f2]).T
#
#        # Mirror phi1 and phi2 around 90 and 270 deg
#        mask1 = Ri[:,0] % 2 == 1
#        mask2 = Ri[:,2] % 2 == 1
#        euler[mask1, 0] = np.pi/2.0 - euler[mask1, 0]
#        euler[mask1, 2] = np.pi/2.0 - euler[mask1, 2]
#        euler[mask2, 0] = np.pi/2.0 - euler[mask2, 0]
#        euler[mask2, 2] = np.pi/2.0 - euler[mask2, 2]
#
#        # Set interval for phi1, phi2
#        euler[:, [0, 2]] += np.pi/2.0 * Ri[:, [0, 2]]
#
#        # Mirror Phi around 90 deg
#        mask = Ri[:, 1] == 1
#        euler[mask, 1] = np.pi - euler[mask, 1]
#
#        return euler * 180. / np.pi







# SUBROUTINES NOT USED FOR ODF CALCULATION
#
#def read_oimfile(fname, sep=None, headermark='# Column '):
#    """Reads a OIM text export file and returns its header, data and
#    comments.
#
#    Arguments
#    ---------
#    fname: string
#        Name of file to read.
#    sep: None | string
#        Data seperator. Use None for ODF data and ';' for MDF and
#        other charts.
#    headermark: string
#        String matching start of header lines defining the columns.
#        Use the default for ODF data and 'Column ' for MDF and other
#        charts.
#
#    Returns
#    -------
#    header: list
#        Column names.
#    data: dict
#        Dict mapping column names to NumPy arrays with column data.
#    comments: dict
#        Dict mapping column names to comment strings.
#    """
#    import re
#    f = open(fname, 'r') if isinstance(fname, str) else fname
#    head = {}
#    comments = {}
#    line = ' '
#    while not line.startswith(headermark):
#        line = f.readline()
#    while line.startswith(headermark):
#        m = re.match('^' + headermark + r'\s*(\d+)(-(\d+))?: (.*)', line)
#        col1, dum, col2, descr = m.groups()
#        col1 = int(col1)
#        cmt = None
#        n = descr.find('(')
#        nn = descr.find('[')
#        if nn > -1 and (nn < n or n == -1):
#            n = nn
#        if n > -1:
#            cmt = descr[n:].strip(' ()[]\n\r\t')
#            descr = descr[:n].strip()
#        if col2:
#            col2 = int(col2)
#            for h in descr.split(','):
#                h = h.strip()
#                head[h] = col1
#                comments[h] = cmt
#                col1 += 1
#            if col1 != col2 + 1:
#                raise RuntimeError('Headers of expected %d columns cannot '
#                                   'be identified while parsing: %r' % (
#                        col2 - col1 + 1, line))
#        else:
#            h = descr.strip()
#            head[h] = col1
#            comments[h] = cmt
#        line = f.readline()
#
#    header = [k for k,v in sorted(head.items(), key=lambda item: item[1])]
#
#    data = dict((k, []) for k in head.keys())
#    while True:
#        if not line:
#            break
#        tokens = line.split(sep)
#        if not tokens[0].strip():  # break on empty tokens...
#            break
#        for i, k in enumerate(header):
#            data[k].append(asnumber(tokens[i]))
#        line = f.readline()
#    for k in data:
#        data[k] = np.array(data[k])
#
#
#    if isinstance(fname, str):
#        f.close()
#    return header, data, comments
#
#
#
#def ori_from_oimfile(fname):
#    """Returns a Orientations object read from OIM text export file *fname*."""
#    header, data, comments = read_oimfile(fname)
#    angles = np.array([data['phi1'], data['PHI'], data['phi2']]).T*180./np.pi
#    name = fname.name if hasattr(fname, 'name') else fname
#    ori = Orientations(angles, label='Read form %r' % name)
#    return ori
#
#
#def mdf_from_oimfile(fname):
#    """Returns MDF data read from OIM text export file *fname*."""
#    header, data, comments = read_oimfile(fname, sep=';', headermark='Column ')
#    return data
#
#
#def get_primes_to(n):
#    """Return all primes up to *n*."""
#    numbers = set(range(n, 1, -1))
#    primes = []
#    while numbers:
#        p = numbers.pop()
#        primes.append(p)
#        numbers.difference_update(set(range(p*2, n+1, p)))
#    return np.array(primes)
#
#def get_first_primes(n, base=100000):
#    """Returns the *n* first primes."""
#    initbase = base
#    while True:
#        primes = get_primes_to(base)
#        if len(primes) >= n:
#            return primes[:n]
#        base += initbase
#
#def halton_number(index, base):
#    """Returns the Halton number index *index* of base *base*."""
#    result = 0
#    f = 1./base
#    i = index
#    while i > 0:
#        result += f * (i % base)
#        i = np.floor(i / base)
#        f /= base
#    return result
#
#def halton(npoints, ndim=1):
#    """Returns *npoints* Halton numbers in *ndim* dimensions."""
#    bases = get_first_primes(ndim)
#    return np.array([[halton_number(i, b) for b in bases]
#                     for i in range(npoints)]).squeeze()
    
    