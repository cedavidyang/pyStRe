class FormResults(object):
    """ postprocessing results of FORM analysis"""
    def __init__(self, iternum, beta1, pf1, dsptu, dsptx, alpha):
        """ results of reliability analysis
            iternum: iteration number
            beta1: reliability index by FORM
            pf1: failure probability by FORM
            dsptu: design point in u space (reduced norm)
            dsptx: design point in x space (random variable)
            alpha: alpha vector
        """
        self.iternum = iternum
        self.beta1 = beta1
        self.pf1 = pf1
        self.dsptu = dsptu
        self.dsptx = dsptx
        self.alpha = alpha
        # sensitivity results
        self.imptg = None
        self.gfunchist = None
        self.gradGhist = None
        self.stephist = None

    def set_sens_res(self, imptg, gfunchist, gradGhist, stephist):
        """sensitivity results
           imptg: importance vector gamma
           gfunchist: recorded gfunc values
           gradGhist: recorded gradient values of gfunc
           stephist: recorded step size values
        """
        self.imptg = imptg
        self.gfunchist = gfunchist
        self.gradGhist = gradGhist
        self.stephist = stephist


class ReliabilityResults(object):
    """ postprocessing results of FORM analysis"""
    def __init__(self, beta, pf):
        """ results of reliability analysis
            beta: reliability
            pf: failure probability
        """
        self.beta = beta
        self.pf = pf
