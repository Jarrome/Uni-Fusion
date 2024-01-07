import torch
def IRLS(y,X,maxiter=100, w=None, IRLS_p = 1, d=0.0001, tolerance=1e-3):
    n,p = X.shape
    delta = torch.ones((1,n),dtype=torch.float64).to(X) * d
    if w is None:
        w = torch.ones((1,n),dtype=torch.float64).to(X)
    #W = torch.diag(w[0,:]) # n,n
    #XTW = X.transpose(0,1).matmul(W)
    XTW = X.transpose(0,1)*w #p,n
    B = XTW.matmul(X).inverse().matmul(XTW.matmul(y))
    for _ in range(maxiter):
        _B = B
        _w = torch.abs(y-X.matmul(B)).transpose(0,1)
        #w = 1./torch.max(delta,_w)
        w = torch.max(delta,_w) ** (IRLS_p-2)
        #W = torch.diag(w[0,:])
        #XTW = X.transpose(0,1).matmul(W)
        XTW = X.transpose(0,1)*w
        B = XTW.matmul(X).inverse().matmul(XTW.matmul(y))
        tol = torch.abs(B-_B).sum()
        if tol < tolerance:
            return B, w
    return B, w
