import sys
import numpy as np
import scipy.stats as stats
import scipy.optimize as op
from probdata import ProbData
from analysisopt import AnalysisOpt
from postprocessing import FormResults, ReliabilityResults
from gfunc import Gfunc


def _search_dir(G, grad_G, u):
    alpha = -grad_G / np.linalg.norm(grad_G)
    d = (G/np.linalg.norm(grad_G)+np.dot(alpha,u))*alpha-u
    return d


def _gen_DS_solver(corrT, ncsrv, tol, ntrial, contol=1e-6, opttol=1e-8):
    """ corrT: target correlation
        ncsrv: number of common source random variables
    """
    nls = corrT.shape[0]
    lb  = np.zeros((nls,ncsrv))
    ub  = np.ones((nls,ncsrv))
    bnds = np.asfarray([lb.flatten(),ub.flatten()]).T
    #initr = 0.5*np.ones((nls,ncsrv))

    def residual_gen_DS(r, corrT=corrT, ncsrv=ncsrv):
        nls = corrT.shape[1]
        rmatrix = np.resize(r,(nls,ncsrv))
        corr = np.dot(rmatrix,rmatrix.T)
        corr = corr-np.diag(np.diag(corr))+np.eye(nls)
        normR = np.linalg.norm(corrT-corr, ord='fro')
        return normR
    def con_gen_DS(r, nls=nls, ncsrv=ncsrv, contol=contol):
        rmatrix = np.resize(r,(nls,ncsrv))
        c = (1.-contol) - np.sum(rmatrix**2,axis=1)
        return c
    def congrad_gen_DS(r, nls=nls, ncsrv=ncsrv):
        rmatrix = np.resize(r,(nls,ncsrv))
        gradc = []
        for icsrv in range(ncsrv):
            gc = -2.*np.diag(rmatrix[:,icsrv])
            gradc.append(gc)
        return np.vstack(gradc).T

    consbnds = [{'type': 'ineq', 'fun':con_gen_DS}]
    for i in xrange(nls*ncsrv):
        lbi = lb.flatten()[i]
        ubi = ub.flatten()[i]
        def ineqi(x,indx=i, lbi=lbi): return x[indx]-lbi
        consbnds.append({'type': 'ineq', 'fun':ineqi})
        def ineqj(x,indx=i, ubi=ubi): return ubi-x[indx]
        consbnds.append({'type': 'ineq', 'fun':ineqj})

    initr0 = 0.5*np.ones((nls,ncsrv))
    initr = initr0.flatten()
    # opres = op.minimize(residual_gen_DS, initr, method='SLSQP', bounds=bnds,
            # constraints={'type':'ineq', 'fun':con_gen_DS, 'jac':congrad_gen_DS},
            # tol=opttol, options={'maxiter':int(1e4), 'disp':False})
    opres = op.minimize(residual_gen_DS, initr, method='COBYLA',
            constraints=consbnds, tol=opttol,
            callback=None, options={'maxiter': int(1e4), 'disp':False})
    r = opres.x
    msg = opres.message
    exitflag = opres.success

    # opres_array = []
    # while fval>tol and itrial<ntrial:
        # indx = np.random.permutation(np.arange(nls*ncsrv))[:int(np.floor(nls*ncsrv/2))]
        # initr = initr0.flatten()
        # initr[indx] = -initr[indx]
        # # opres = op.minimize(residual_gen_DS, initr, args=(corrT, ncsrv), bounds=bnds,
                # # #constraints={'type':'ineq', 'fun':con_gen_DS, 'jac':congrad_gen_DS, 'args':(nls,ncsrv)}, tol=1e-16,
                # # constraints={'type':'ineq', 'fun':con_gen_DS, 'args':(nls,ncsrv)}, tol=opttol,
                # # options={'maxiter':int(1e4), 'disp':False})
        # opres = op.minimize(residual_gen_DS, initr, args=(corrT, ncsrv), bounds=bnds,
                # constraints={'type':'ineq', 'fun':con_gen_DS, 'jac':congrad_gen_DS, 'args':(nls,ncsrv)},
                # tol=opttol, options={'maxiter':int(1e4), 'disp':False})
        # fval = opres.fun
        # if np.isnan(fval):
            # opres.x = np.zeros((nls,ncsrv))
            # opres.fun = 1.0
        # opres_array.append(opres)
        # itrial += 1
    # fvals = [opres.fun for opres in opres_array]
    # indx = np.nanargmin(fvals)
    # r = opres_array[indx].x
    # msg = opres_array[indx].message
    # exitflag = opres_array[indx].success

    rmatrix = np.resize(r,(nls,ncsrv))
    rescheck=np.sum(rmatrix**2,axis=1)
    if any(rescheck>1):
        rng = rmatrix[rescheck>1,:]
        numpow = np.abs(np.fix(np.log10(np.sum(rng**2,axis=1,keepdims=True)-1.)))-1.
        rnew = np.fix(rng*(10.**numpow*np.ones((1,rng.shape[1]))))\
            /(10.**numpow*np.ones((1,rng.shape[1])))
        rmatrix[rescheck>1,:] = rnew
        rescheck=np.sum(rmatrix**2,axis=1)

    corrDS = np.dot(rmatrix,rmatrix.T)
    corrDS = corrDS-np.diag(np.diag(corrDS))+np.eye(nls)
    normerr = np.linalg.norm(corrDS-corrT)
    iterres =[normerr,msg,exitflag]

    return rmatrix, corrDS, rescheck, iterres


def _prob_vector(pmrg):
    for i in xrange(np.size(pmrg)):
        if i == 0:
            p = np.array([pmrg[i], 1.-pmrg[i]])
        else:
            p = np.hstack((p*pmrg[i], p*(1.-pmrg[i])))
    return p


def _event_matrix(nevent):
    for i in xrange(nevent):
        if i==0:
            C = np.array([[1.], [0.]])
        else:
            lenC = np.max(np.shape(C))
            C1 = np.hstack( (C, np.ones((lenC,1))) )
            C2 = np.hstack( (C,np.zeros((lenC,1))) )
            C = np.vstack((C1,C2))
    return C


def _p_vector_mod(pmarg, compn):
    for i,pi in enumerate(pmarg):
        if i == compn-1:
            pComp = 1.; pCompComp = -1.
        else:
            pComp = pi; pCompComp = 1.-pi
        if i == 0:
            p = np.hstack((pComp, pCompComp))
        else:
            p = np.hstack((p*pComp, p*pCompComp))

    return p


class CompReliab(object):

    def __init__(self, probdata, gfunc, analysisopt=None):
        if analysisopt is None:
            analysisopt = AnalysisOpt()

        self.probdata = probdata
        self.gfunc = gfunc
        self.analysisopt = analysisopt
        self.result = None


    def _step_size_multiproc(self, G, grad_G, u, d, ntrial=6):
        """u: current point in reduced normal space
           d: search direction
        """
        c = 2*(np.linalg.norm(u)/np.linalg.norm(grad_G))+10
        merit = 0.5*(np.linalg.norm(u))**2+c*abs(G)

        for i in range(ntrial):
            trialStepSize = 0.5**i
            trialu = u+trialStepSize*d
            trialx = self.probdata.u_to_x(trialu)
            trialG = self.gfunc.g_value(trialx)
            meritnew = 0.5*(np.linalg.norm(trialu))**2+c*abs(trialG)
            if meritnew<=merit:
                break

        if self.analysisopt.verbose and i==ntrial:
            print 'The step size has been reduced by a factor of 1/{:d} before continuing.'.format(2**ntrial)

        return trialStepSize

    def form_result(self):
        nrv = np.size(self.probdata.rvs)
        verbose = self.analysisopt.verbose
        e1 = self.analysisopt.e1
        e2 = self.analysisopt.e2
        crGhist = []
        crUhist = []
        gfunchist = []
        stephist = []
        if self.analysisopt.recordu:
            uhist = []
        if self.analysisopt.recordx:
            xhist = []
        if self.analysisopt.flagsens:
            gradGhist = []
            alphahist = []
            imptghist = []    #parameter importance history
        # Nataf transformation of correlation matrix and Cholesky decomposition
        self.probdata.cholesky(self.analysisopt.flagsens, self.analysisopt.verbose)
        # Compute starting point for the algorithm
        x = self.probdata.startpoint
        u = self.probdata.x_to_u(x)

        # Set parameters for the iterative loop
        i = 0          # Initialize counter
        conv_flag = False  # Convergence is achieved when this flag is set to 1
        Go = self.gfunc.g_value(x)
        if verbose:
            print 'Value of limit-state function in the first step: {:f}'.format(Go)

        while conv_flag is False:
            if verbose:
                print '..................................'
                print 'Now carrying out iteration number: {:d}'.format(i+1)

            # Transformation from u to x space
            x = self.probdata.u_to_x(u)
            J_u_x = self.probdata.jacobian(x,u)
            J_x_u = np.linalg.inv(J_u_x)
            # Evaluate limit-state function and its gradient
            G = self.gfunc.g_value(x)
            grad_g = self.gfunc.dgdq_vale(x)

            grad_G = np.dot(grad_g, J_x_u)
            gfunchist.append(G)

            # Set scale parameter Go and inform about struct. resp.

            # Compute alpha vector
            alpha = -grad_G / np.linalg.norm(grad_G)   # Alpha vector
            # Compute gamma vector
            D_prime = np.diag(np.diag(np.sqrt(np.dot(J_x_u, J_x_u.T))))     # (intermediate computation)
            imptg = np.dot(alpha,np.dot(J_u_x,D_prime)) / \
                    np.linalg.norm(np.dot(alpha,np.dot(J_u_x,D_prime)))   # Importance vector gamma

            # Check convergence
            cr2 = np.linalg.norm( u-np.dot(alpha, np.dot(u,alpha)) )
            if ( (abs(G/Go)<e1) and (cr2<e2) ) or i==self.analysisopt.imax-1:
                conv_flag = True

            crGhist.append( abs(G/Go) )
            crUhist.append( cr2 )
            if self.analysisopt.recordu:
                uhist.append(u)
            if self.analysisopt.recordx:
                xhist.append(x)

            if self.analysisopt.flagsens:
                gradGhist.append(grad_G)  # Value of grad_G(u)
                alphahist.append(alpha)
                imptghist.append(imptg)

            # Take a step if convergence is not achieved
            if not conv_flag:

                # Determine search direction
                d = _search_dir(G,grad_G,u)
                # Determine step size
                if self.analysisopt.stepcode == 0:
                    step = self._step_size_multiproc(G,grad_G,u,d)
                else:
                    step = self.analysisopt.stepcode

                stephist.append(step)

                # Determine new trial point
                unew = u + step * d;

                # Prepare for a new round in the loop
                u = unew
                i = i + 1


        if i == self.analysisopt.imax:
            print 'Maximum number of iterations was reached before convergence.'
        else:
            # Post-processing
            beta1 = np.dot(alpha,u)
            pf1= stats.norm.cdf(-beta1)
            formresults = FormResults(i, beta1, pf1, u, x, alpha)

            if self.analysisopt.flagsens:
                D_prime = np.diag(np.diag(np.sqrt(np.dot(J_x_u,J_x_u.T))))
                imptg = np.dot(np.dot(alpha, J_u_x),D_prime)/\
                        np.norm(np.dot(np.dot(alpha,J_u_x),D_prime))
                formresults.set_sens_res(imptg,gfunchist, gradGhist, stephist)

            self.result = formresults

        return formresults


class SysReliab(object):

    def __init__(self, sysdef, comps=None, beta=None, syscorr=None, rvnames=None):
        """ comps should be a list object."""

        # system definition
        if np.size(sysdef) == 1:
            if sysdef[0]<0:
                self.systype = 'series'
            elif sysdef[0]>0:
                self.systype = 'parallel'
            else:
                print "sysdef with only one zero is not permitted"
        else:
            self.systype = 'general'

        if comps is None:
            if (beta is None) or (syscorr is None) or (rvnames is None):
                print "ERROR in SysReliab: must provide beta, syscorr, rvnames"
                sys.exit(1)
            self.sysdef = sysdef
            self.rvnames = rvnames
            self.beta=beta
            self.syscorr = syscorr
        else:
            self.comps = comps
            self.sysdef = sysdef
            self.rvnames = []
            # rv names of all limit states
            for comp in comps:
                cmpNames = comp.probdata.names
                for name in cmpNames:
                    if name not in self.rvnames:
                        self.rvnames.append(name)
            self._setup_sys()

        self.nCSrv = None
        self.rmtx = None
        if self.sysdef[1].lower() == 'event':
            ncomp = np.size(self.sysdef[0])
        else:
            ncomp = np.max(np.abs(self.sysdef[0]))
        C = _event_matrix(ncomp)
        self._sys_event(C)


    def _sys_event(self, C):
        sysdef = self.sysdef
        row,col = C.shape

        if self.systype.lower() == 'series':
            self.cutset = 1-np.prod(np.ones((row,col))-C, axis=1, dtype=int)
            ncutsets = []
        elif self.systype.lower() == 'parallel':
            self.cutset = np.prod(C, axis=1, dtype=int)
            ncutsets = []
        else:
            syscode = np.array(sysdef[0])
            systype = sysdef[1]
            if systype.lower() == 'event':
                allevent = C
                fpattern = syscode
                self.cutset = np.all(allevent==fpattern, axis=1).astype(float)
            else:
                if systype.lower() == 'link':
                    syscode = -syscode
                syscode = np.hstack((0,syscode,0))
                sysnonzero = np.where(syscode!=0)[0]
                syszero = np.where(syscode==0)[0]
                int1 = syszero-np.hstack((0,syszero[:-1]))
                sizeCutSets = int1[int1>1]-1
                ncutsets = np.max(sizeCutSets.shape)
                cCutSets = np.zeros((row, ncutsets))
                for i in xrange(ncutsets):
                    cCutSet = np.ones(row)
                    for j in xrange(sizeCutSets[i]):
                        comp = syscode[sysnonzero[np.sum(sizeCutSets[:i])+j]]
                        if comp<0:
                            cCutSet = cCutSet*(np.ones((row,1))-C[:,abs(comp)-1])
                        else:
                            cCutSet = cCutSet*C[:,comp-1]
                    cCutSets[:,i] = cCutSet
                self.cutset = np.ones(row) - np.prod(np.ones((row,ncutsets))-cCutSets, axis=1)


    def _expand_alpha(self, cmpNames, cmpAlpha):
        alphaall = np.zeros(np.size(self.rvnames))
        for name,alphai in zip(cmpNames,cmpAlpha):
            indx = np.where(np.array(self.rvnames) == name)[0][0]
            alphaall[indx] = alphai
        return alphaall


    def _setup_sys(self):
        ncomp = np.size(self.comps)
        nrv = np.size(self.rvnames)
        beta = np.zeros(ncomp)
        alpha = np.zeros((ncomp, nrv))
        for i, comp in enumerate(self.comps):
            if comp.result is None:
                formresults = comp.form_result()
            else:
                formresults = comp.result
            beta[i] = formresults.beta1
            alpha[i,:] = self._expand_alpha(comp.probdata.names, formresults.alpha)
        self.beta = beta
        self.alpha = alpha
        R = np.dot(alpha,alpha.T)  # alpha is a matrix with alpha of the i-th limit state on the i-th column
        R=R-np.diag(np.diag(R))+np.eye(R.shape[0])
        self.syscorr = R

    def update_setup_sys(self, compindx, newcomps):
        for i in compindx:
            self.comps[i] = newcomps[i]
            comp = self.comps[i]
            formresults = comp.form_result()
            self.beta[i] = formresults.beta1
            sefl.alpha[i,:] = self._expand_alpha(comp.probdata.names, formresults.alpha)
        beta = self.beta
        alpha = self.alpha
        R = np.dot(alpha,alpha.T)  # alpha is a matrix with alpha of the i-th limit state on the i-th column
        R=R-np.diag(np.diag(R))+np.eye(R.shape[0])
        self.syscorr = R

    def set_nCSrv(self, nCSrv=None, tol=1e-6, nmax=10, ntrial=5):
        if nCSrv is None:
            nCSrv = 1
            itercr = 1.0
            while itercr>tol and nCSrv<=nmax:
                r, corrDS, rescheck, iterres = _gen_DS_solver(self.syscorr, nCSrv, tol, ntrial)
                itercr = iterres[0]
                nCSrv += 1
            nCSrv -= 1
        else:
            r, corrDS, rescheck, iterres = _gen_DS_solver(self.syscorr, nCSrv, tol, ntrial)
        self.nCSrv, self.rmtx, self.syscorrDS = nCSrv, r, corrDS


    def form_msr(self, analysisopt=None):
        systype = self.systype
        cutset = self.cutset
        beta = self.beta
        if (self.nCSrv is not None) and (self.rmtx is not None):
            ncsrv = self.nCSrv
            r = self.rmtx
        else:
            print('assign common source random variables first')
            sys.exit(1)
        # create probdata and analysisopt
        names = [str(e) for e in range(ncsrv+1)]
        rvs = [stats.norm() for i in range(ncsrv+1)]
        corr = np.eye(ncsrv+1)
        probdata = ProbData(names=names, rvs=rvs, corr=corr, nataf=False)
        if analysisopt is None:
            analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False,
                    flagsens=False, verbose=False)

        def gf_dgf(s, cutset, systype, beta, r, param=None):
            pmarg = stats.norm.cdf( (-beta-np.dot(r,s[:-1]))/np.sqrt(1.-np.sum(r**2,axis=1)) )
            dPds = -np.diag(stats.norm.pdf((-beta-r.dot(s[:-1]))/np.sqrt(1-np.sum(r**2,axis=1))) /
                        np.sqrt(1-np.sum(r**2,axis=1))).dot(r)
            if systype.lower() == 'general':
                p = _prob_vector(pmarg)
                gf = s[-1] - stats.norm.ppf(cutset.dot(p))
                ph = np.zeros((2**(beta.size), beta.size))
                for i,betai in enumerate(beta):
                    pMargMod = _p_vector_mod(pmarg,i+1)
                    ph[:,i] = pMargMod
                dpds = ph.dot(dPds)
                dgf = np.hstack((-cutset.dot(dpds).dot(1/(stats.norm.pdf(s[-1]-gf))), 1))
            elif systype.lower() == 'parallel':
                gf = s[-1] - stats.norm.ppf(np.prod(pmarg))
                dgf = np.hstack((np.dot(-np.prod(pmarg)*(1./pmarg).T,dPds)/stats.norm.pdf(s[-1]-gf),1.))
            elif systype.lower() == 'series':
                gf = -( s[-1] - stats.norm.ppf(np.prod(1.-pmarg)) )
                dgf = np.hstack((np.dot(-np.prod(1.-pmarg)*(1./(1.-pmarg)).T,dPds)/stats.norm.pdf(s[-1]-gf),-1.))
            else:
                print('unknown system type'); sys.exit(1)
            return gf,dgf

        param=None
        gf = lambda x,param: gf_dgf(x, cutset, systype, beta, r, param)[0]
        dgf = lambda x,param: gf_dgf(x, cutset, systype, beta, r, param)[1]
        gfunc = Gfunc(gf,dgf)

        formBeta = CompReliab(probdata, gfunc, analysisopt)
        formresults = formBeta.form_result()
        #if systype.lower() in ['series','general']:
            #formresults.alpha = -formresults.alpha
            #formresults.pf1 = 1.-formresults.pf1
            #if formresults.imptg is not None:
                #formresults.imptg = -formresults.imptg

        return formresults


    def direct_msr(self, tol=1e-7, intLb=-6, intUb=6):
        def integrnd(beta, r, cutset, systype, *param):
            nparam = len(param)
            s = param
            ps = stats.norm.cdf((-beta-np.dot(r,s))/np.sqrt(1-np.sum(r**2, axis=-1)))
            if systype.lower() == 'parallel':
                intval = np.prod(ps)*stats.multivariate_normal.pdf(s,mean=None,cov=np.eye(nparam))
            elif systype.lower() == 'series':
                intval = np.prod(1-ps)*stats.multivariate_normal.pdf(s,mean=None,cov=np.eye(nparam))
            if systype.lower() == 'general':
                p = _prob_vector(ps)
                intval = cutset.dot(p)*stats.multivariate_normal.pdf(s,mean=None,cov=np.eye(nparam))
            return intval
        systype = self.systype
        cutset = self.cutset
        beta = self.beta
        if (self.nCSrv is not None) and (self.rmtx is not None):
            ncsrv = self.nCSrv
            r = self.rmtx
        else:
            print('assign common source random variables first')
            sys.exit(1)
        import scipy.integrate as intg
        if self.nCSrv==1:
            intsol = intg.quad(lambda x: integrnd(beta, r, cutset, systype, x), intLb, intUb, epsabs=tol)
        elif self.nCSrv==2:
            intsol = intg.dblquad(lambda x,y: integrnd(beta, r, cutset, systype, x,y), intLb, intUb,
                    lambda x: intLb, lambda x: intUb, epsabs=tol)
        elif self.nCSrv==3:
            intsol = intg.tplquad(lambda x,y,z: integrnd(beta, r, cutset, systype, x,y,z), intLb, intUb,
                    lambda x: intLb, lambda x: intUb,
                    lambda x,y: intLb, lambda x,y: intUb, epsabs=tol)
        else:
            print 'Direct integration does not support nCSrv>3'
            sys.exit(1)

        if systype == 'series':
            syspf = 1.-intsol[0]
        else:
            syspf = intsol[0]
        sysbeta = stats.norm.ppf(1-syspf)
        results = ReliabilityResults(sysbeta, syspf)

        return results

    def mvn_msr(self, corrDS=None, abstol=1e-12, reltol=1e-12, intLb=-10, intUb=10):
        systype = self.systype
        beta = self.beta
        nls = len(self.rvnames)
        if corrDS is None:
            correl = self.syscorrDS[np.triu_indices(nls,1)]
        else:
            correl = corrDS[np.triu_indices(nls,1)]
        if corrDS is None:
            corrDS = self.syscorrDS
        i=1; n=10000; syspf0=0.; dpf=1.
        # while i!=0:
            # n +=10000
            # v,res,i = stats.mvn.mvndst(intLb*np.ones(nls), beta, np.zeros(nls, dtype=int), correl, [nls*n,1e-12, 1e-12])
        while i!=0:
            n+=10000
            res,i = stats.mvn.mvnun(-10*np.ones(nls), beta, np.zeros(nls), corrDS, [nls*n, abstol, reltol])
        # if abs(res-res1)/(0.5*(res+res1))>1e-3:
            # print 'warning: abnormal difference between mvnun and mvndst results'
        if systype.lower() == 'series':
            syspf = 1.-res
            sysbeta = -stats.norm.ppf(syspf)
            results = ReliabilityResults(sysbeta, syspf)
        elif systype.lower() == 'parallel':
            syspf = res
            sysbeta = -stats.norm.ppf(syspf)
            results = ReliabilityResults(sysbeta, syspf)
        else:
            print('mvn_msr only supports series or parallel system')
            sys.exit(0)

        return results


if __name__ == '__main__':
    from probdata import ProbData
    from analysisopt import AnalysisOpt
    from gfunc import Gfunc

    import scipy.stats as stats

    # uncorrelated case in example 5.11 in Nowak's book (2013)
    Z = stats.norm(loc=100.0, scale=100.0*0.04)
    Fy = stats.lognorm(np.sqrt(np.log(1+0.1**2)), scale=40.0/np.sqrt(1+0.1**2))
    M = stats.gumbel_r(loc=2000-np.sqrt(6)*200/np.pi*np.euler_gamma,
            scale=np.sqrt(6)*200/np.pi)
    rvs = [Z, Fy, M]
    corr = np.array([[1., 0.9, 0.], [0.9, 1., 0.], [0., 0., 1.]])
    probdata = ProbData(names=['Z','Fy','M'], rvs=rvs, corr=corr, startpoint=[100.,40.,2000.],
            nataf=False)

    def gf1(x, param=None):
        return x[0]*x[1]-x[2]
    def dgdq1(x, param=None):
        dgd1 = x[1]
        dgd2 = x[0]
        dgd3 = -1.
        return [dgd1,dgd2,dgd3]

    analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc = Gfunc(gf1, dgdq1)

    formBeta = CompReliab(probdata, gfunc, analysisopt)
    formresults = formBeta.form_result()
    beta = formresults.beta1

    # a series system with two independent ex5.11 problems
    import copy
    formBeta2 = copy.deepcopy(formBeta)
    formBeta2.probdata.names = ['Z2', 'Fy2', 'M2']
    sysBeta = SysReliab([formBeta, formBeta2], [2])
    sysBeta.set_nCSrv()
    sysformres = sysBeta.form_msr()
