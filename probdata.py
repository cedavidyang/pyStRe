import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad, fixed_quad
import scipy.optimize as op

def _dblquad0(rvi,rvj,rho0,ns):
    """ Computes z1, z2 and x1, x2 values for Gauss integration - Vectorized version"""
    stnorm = stats.norm()
    mi, vi = rvi.stats(); si = np.sqrt(vi)
    mj, vj = rvj.stats(); sj = np.sqrt(vj)
    res = np.zeros(np.asarray(rho0).shape)
    for i,rhoij in enumerate(rho0):
        mulnorm = stats.multivariate_normal([0.,0.], [[1.,rhoij], [rhoij,1.]])
        # integration function
        def fij(yi,yj):
            zi = (rvi.ppf(stnorm.cdf(yi))-mi)/si
            zj = (rvj.ppf(stnorm.cdf(yj))-mj)/sj
            return zi*zj*mulnorm.pdf([yi,yj])

        quadres = dblquad(fij, -ns, ns, lambda x: -ns, lambda x: ns)
        res[i] = quadres[0]

    return res

def _dblquad(rvi,rvj,rho0,ns,nip):
    """ Computes z1, z2 and x1, x2 values for Gauss integration - Vectorized version"""
    stnorm = stats.norm()
    mi, vi = rvi.stats(); si = np.sqrt(vi)
    mj, vj = rvj.stats(); sj = np.sqrt(vj)
    res = np.zeros(np.asarray(rho0).shape)
    for i,rhoij in enumerate(rho0):
        mulnorm = stats.multivariate_normal([0.,0.], [[1.,rhoij], [rhoij,1.]])
        # integration function
        def fij(yi,yj):
            zi = (rvi.ppf(stnorm.cdf(yi))-mi)/si
            zj = (rvj.ppf(stnorm.cdf(yj))-mj)/sj
            y = np.zeros(np.asarray(yi).shape+(2,))
            y[:,0] = yi
            y[:,1] = yj
            return zi*zj*mulnorm.pdf(y)
        def fj(yj):
            return fixed_quad(fij, -ns, ns, args=(yj,), n=nip)[0]

        quadres = fixed_quad(fj, -ns, ns, n=nip)
        res[i] = quadres[0]

    return res


class ProbData(object):
    """ probdata class, equivalent to probdata struct in FERUM

    Attributes:
      name: rv name
      rvs: list of random variable objects
      corr: correlation of random variables
      startpoint: start point of FORM
    """
    def __init__(self, names, rvs, corr=None, startpoint=None, nataf=False):
        self.names = names
        self.rvs = rvs
        if corr is None:
            self.corr = np.eye(np.size(rvs))
        else:
            self.corr = corr
        if startpoint is None:
            self.startpoint = np.asarray([rv.mean() for rv in rvs])
        else:
            self.startpoint = startpoint
        self.lo = None
        self.ilo = None
        self.nataf = nataf

    def setcorr(self, corr):
        self.corr = corr

    def _mod_corr_solve(self, flagsens, verbose, tol=1e-5, ns=5, nip=32):
        nrv = np.size(self.rvs)
        if flagsens:
            dRodrho           = np.eye((nrv,nrv))

            dRodthetafi.mu    = np.zeros((nrv,nrv))
            dRodthetafi.sigma = np.zeros((nrv,nrv))
            dRodthetafi.p1    = np.zeros((nrv,nrv))
            dRodthetafi.p2    = np.zeros((nrv,nrv))
            dRodthetafi.p3    = np.zeros((nrv,nrv))
            dRodthetafi.p4    = np.zeros((nrv,nrv))

            dRodthetafj.mu    = np.zeros((nrv,nrv))
            dRodthetafj.sigma = np.zeros((nrv,nrv))
            dRodthetafj.p1    = np.zeros((nrv,nrv))
            dRodthetafj.p2    = np.zeros((nrv,nrv))
            dRodthetafj.p3    = np.zeros((nrv,nrv))
            dRodthetafj.p4    = np.zeros((nrv,nrv))
        else:
            dRodrho = None
            dRodthetafi = None
            dRodthetafj = None

        # loop over all off diagonal element of correlation matrix
        corrnew = np.ones((nrv,nrv))
        for i in range(nrv):
            for j in range(i+1, nrv):
                rvi = self.rvs[i]
                rvj = self.rvs[j]
                rho = self.corr[i,j]
                mi, vi = rvi.stats(); si = np.sqrt(vi)
                mj, vj = rvj.stats(); sj = np.sqrt(vj)
                ibound = np.array([mi-ns*si, mi+ns*si])
                jbound = np.array([mj-ns*sj, mj+ns*sj])
                def objfunc(x):
                    return abs(_dblquad(rvi, rvj, x, ns)-rho)
                #opres = op.minimize_scalar(objfunc, bounds=[-1.0+tol, 1.0-tol], method='Bounded')
                test = _dblquad(rvi, rvj, np.array([0.9]), ns, nip=nip)
                opres = op.minimize(objfunc, rho, bounds=((-1.0+tol, 1.0-tol),))
                rho0 = opres.x
                corrnew[i,j] = rho0
        for i in range(nrv):
            for j in range(i+1, nrv):
                corrnew[j,i] = corrnew[i,j]

        return corrnew, dRodrho, dRodthetafi, dRodthetafj


    def cholesky(self, flagsens, verbose):
        # Nataf transformation of correlation matrix
        if self.nataf:
            if verbose:
                print '\n'
                print 'Computation of modified correlation matrix R0'
                print 'Takes some time if sensitivities are to be computed with gamma (3), beta (7) or chi-square (8) distributions.'
                print 'Please wait... (Ctrl+C breaks)'
            [corrnew, dRodrho, dRodthetafi, dRodthetafj] = self._mod_corr_solve(flagsens, verbose)
            self.corr = corrnew

        # Cholesky decomposition
        try:
            lo = np.linalg.cholesky(self.corr)
        except LinAlgError:
            print "error in probdata.py cholesky(): self.corr must be positive definitive"
            sys.exit(1)
        self.lo = lo
        self.ilo = np.linalg.inv(lo)

    def x_to_u(self, x):
        if self.ilo is None:
            print "conduct cholesky depcomposition first"
            sys.exit(1)
        else:
            z = np.copy(x)
            for i, rv in enumerate(self.rvs):
                z[i] = stats.norm.ppf(rv.cdf(x[i]))
            u = self.ilo.dot(z)
        return u

    def u_to_x(self, u):
        if self.lo is None:
            print "conduct cholesky depcomposition first"
            sys.exit(1)
        else:
            z = np.dot(self.lo, u)
            x = np.copy(z)
            for i, rv in enumerate(self.rvs):
                x[i] = rv.ppf(stats.norm.cdf(z[i]))
        return x

    def jacobian(self, x, u):
        if self.lo is None or self.ilo is None:
            print "conduct cholesky depcomposition first"
            sys.exit(1)
        else:
            nrv = np.size(self.rvs)
            z = np.dot(self.lo, u)
            J_z_x = np.zeros((nrv,nrv))

            for i,rv in enumerate(self.rvs):
                pdf1 = rv.pdf(x[i])
                pdf2 = stats.norm.pdf(z[i])
                J_z_x[i,i] = pdf1/pdf2

            J_u_x = np.dot(self.ilo, J_z_x)

        return J_u_x
