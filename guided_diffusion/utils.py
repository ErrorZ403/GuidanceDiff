import torch

def CG(A, b, x, n_inner=5, eps=1e-5):
    r = b - A(x)
    p = r.clone()
    rsold = torch.matmul(r.view(1, -1), r.view(1, -1).T)

    for i in range(n_inner):
        Ap = A(p)
        a = rsold / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

        x = x + a * p
        r = r - a * Ap

        rsnew = torch.matmul(r.view(1, -1), r.view(1, -1).T)
        if torch.abs(torch.sqrt(rsnew)) < eps:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x