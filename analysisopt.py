class AnalysisOpt(object):
    """ analysis option, equivalent to analysisopt struct in FERUM. only FORM related parameters are included

    Attributes:
      ig_max: Maximum number of global iterations allowed in the search algorithm
      il_max: Maximum number of line iterations allowed in the search algorithm
      e1: Tolerance on how close design point is to limit-state surface
      e2: Tolerance on how accurately the gradient points towards the origin
      step_code: 0: step size by Armijo rule, otherwise: given value (0 < s <= 1) is the step size.
      grad_flag: 'DDM': direct differentiation, 'FFD': forward finite difference
    """

    def __init__(self, imax=1000, e1=1e-3, e2=1e-3, stepcode=0,
            gradflag='FFD', ffdparam=1000,ffdparamthetag=1000,
            recordu=True, recordx=True, flagsens=True, verbose=True):
        self.imax = imax
        self.e1 = e1
        self.e2 = e2
        self.stepcode = stepcode
        self.gradflag = gradflag
        self.ffdparam=ffdparam
        self.ffdparamthetag=ffdparamthetag
        self.recordu=recordu
        self.recordx=recordx
        self.flagsens = flagsens
        self.verbose = verbose
