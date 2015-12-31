class Gfunc(object):
    def __init__(self, gfunc, dgdq=None, dgthetag=None, evaluator='basic', param=None,
            gftype='userfunc'):
        self.evaluator = evaluator
        self.param = param
        self.gftype = gftype
        self.dgdq = dgdq
        self.dgthetag = dgthetag
        self.gfunc = gfunc

    def setparam(self, param):
        self.param = param

    def g_value(self, x, param=None):
        if self.gftype == 'userfunc':
            if param is None:
                return self.gfunc(x, self.param)
            else:
                return self.gfunc(x, param)
        else:
            return None

    def dgdq_vale(self, x, param=None):
        if self.gftype == 'userfunc':
            if self.dgdq is None:
                return None
            else:
                if param is None:
                    return self.dgdq(x, self.param)
                else:
                    return self.dgdq(x, param)
        else:
            return None


if __name__ == '__main__':
    def gf1(x, param=None):
        return x[0]-x[1]
    def dg1dq(x, param=None):
        return [1.,-1.]
    gfunc = Gfunc(gf1, dgdq=dg1dq)
    print gfunc.g_value([3,1]), gfunc.dgdq_vale([3,1])
