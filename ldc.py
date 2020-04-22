import numpy as np
from numba import jit
import timeit
import scipy.sparse as sp

@jit(nopython=True)
def thomas(a,b,c,d,x):
    """
    Computes the solution of a system of linear equations of form Ax = d where
    A is a tridiagonal matrix, using Thomas Algorithm.
    
    Parameters
    ==========
    For a system of n linear equations of form Ax = d where,
        A = [[b[0] c[0]   0    0    0    0    0    ...   0],
             [a[0] b[1] c[1]   0    0    0    0    ...   0],
             [  0  a[1] b[2] c[2]   0    0    0    ...   0], ... ,
             [  0    0    0    0    0 a[n-3] b[n-2] c[n-2]],
             [  0    0    0    0    0    0   a[n-2] b[n-1]]]
    
    a : numpy.ndarray of length n-1 
        The elements of the diagonal at offset -1 (below the main diagonal)
    b : numpy.ndarray of length n
        The elements of the main diagonal
    c : numpy.ndarray of length n-1 
        The elements of the diagonal at offset 1 (above the main diagonal)
    d : numpy.ndarray of length n 
        The elements of the right hand side
    x : numpy.ndarray of length n
        An initial guess of the solution
    
    Returns
    =======
    x : numpy.ndarray of length n
        Solution of the given system of equations
    """ 
    n = len(d)
    for i in range(1,n):
        w = a[i-1]/b[i-1]
        b[i] = b[i] - w*c[i-1]
        d[i] = d[i] - w*d[i-1]
    x[n-1] = d[n-1]/b[n-1]
    for i in range(n-2,-1,-1):
        x[i] = (d[i] - c[i]*x[i+1])/b[i]
    return x

def timestep(u,v,w):
    """
    Computes the timestep based on the stability limit, CFL <= 1.2
    
    Parameters
    ==========
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    
    Returns
    =======
    dt : numpy.float64
         Timestep based on CFL criterion, CFL <= 1.2
    """
    umax = np.max(u)
    vmax = np.max(v)
    wmax = np.max(w)
    delt = []
    c = 1.2
    for value, h in zip([umax,vmax,wmax],[hx,hy,hz]):
        if value != 0:
            delt.append(c*h/value)
    dt = np.min(delt)
    return dt

@jit(nopython=True, fastmath=True)
def updateBC(u,v,w,p):
    """
    Updates the boundary values based on the values at the interior points for
    U-velocity, V-velocity, W-velocity and pressure fields.
    
    Parameters
    ==========
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid
    
    Returns
    =======
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
        with updated boundary values
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
        with updated boundary values
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
        with updated boundary values
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid
        with updated boundary values
    """
    # No-slip at x-min and x-max
    for j in range(Ny+1):
        for k in range(Nz+1):
            u[0,j,k] = 0
            u[Nx-1,j,k] = 0
            p[0,j,k] = p[1,j,k]
            p[Nx,j,k] = p[Nx-1,j,k]
            if j != Ny:
                v[0,j,k] = - v[1,j,k]
                v[Nx,j,k] = - v[Nx-1,j,k]
            if k != Nz:
                w[0,j,k] = - w[1,j,k]
                w[Nx,j,k] = - w[Nx-1,j,k]
    # Periodic in y
    for i in range(Nx+1):
        for k in range(Nz+1):
            v[i,0,k] = v[i,Ny-2,k]
            v[i,Ny-1,k] = v[i,1,k]
            p[i,0,k] = p[i,Ny-1,k]
            p[i,Ny,k] = p[i,1,k]
            if i != Nx:
                u[i,0,k] = u[i,Ny-1,k]
                u[i,Ny,k] = u[i,1,k]
            if k != Nz:
                w[i,0,k] = w[i,Ny-1,k]
                w[i,Ny,k] = w[i,1,k]
    # No-slip at z-min and all-fixed at z-max
    for i in range(Nx+1):
        for j in range(Ny+1):
            w[i,j,0] = 0
            w[i,j,Nz-1] = 0
            p[i,j,0] = p[i,j,1]
            p[i,j,Nz] = p[i,j,Nz-1]
            if i != Nx:
                u[i,j,0] = -u[i,j,1]
                u[i,j,Nz] = 2 - u[i,j,Nz-1]
            if j != Ny:
                v[i,j,0] = - v[i,j,1]
                v[i,j,Nz] = - v[i,j,Nz-1]
    
    return u, v, w, p

@jit(nopython=True, fastmath=True)
def equatebc(ustar, vstar, wstar, pstar, u, v, w, p):
    """
    Updates the boundary values of the predicted U-velocity (Ustar),
    V-velocity (Vstar), W-velocity (Wstar) and pressure fields (pstar) 
    based on the values at the interior points for initial
    U-velocity, V-velocity, W-velocity and pressure fields.
    
    Parameters
    ==========
    ustar : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of predicted U-velocity at all the points of 
        the U-velocity grid whose boundary values are to be updated
    vstar : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of predicted V-velocity at all the points of
        the V-velocity grid whose boundary values are to be updated
    wstar : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of predicted W-velocity at all the points of
        the W-velocity grid whose boundary values are to be updated
    pstar : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of predicted pressure at all the points of
        the pressure grid whose boundary values are to be updated
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of initial U-velocity at all the points of
        the U-velocity grid using which boundary values are to be updated
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of initial V-velocity at all the points of
        the V-velocity grid using which boundary values are to be updated
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of initial W-velocity at all the points of
        the W-velocity grid using which boundary values are to be updated
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of initial pressure at all the points of
        the pressure grid using which boundary values are to be updated
    
    Returns
    =======
    ustar : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of predicted U-velocity at all the points of the U-velocity grid
        with updated boundary values
    vstar : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of predicted V-velocity at all the points of the V-velocity grid
        with updated boundary values
    wstar : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of predicted W-velocity at all the points of the W-velocity grid
        with updated boundary values
    pstar : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of predicted pressure at all the points of the pressure grid
        with updated boundary values
    """
    # No-slip at x-min and x-max
    for j in range(Ny+1):
        for k in range(Nz+1):
            ustar[0,j,k] = 0
            ustar[Nx-1,j,k] = 0
            pstar[0,j,k] = p[1,j,k]
            pstar[Nx,j,k] = p[Nx-1,j,k]
            if j != Ny:
                vstar[0,j,k] = - v[1,j,k]
                vstar[Nx,j,k] = - v[Nx-1,j,k]
            if k != Nz:
                wstar[0,j,k] = - w[1,j,k]
                wstar[Nx,j,k] = - w[Nx-1,j,k]
    # Periodic in y
    for i in range(Nx+1):
        for k in range(Nz+1):
            vstar[i,0,k] = v[i,Ny-2,k]
            vstar[i,Ny-1,k] = v[i,1,k]
            pstar[i,0,k] = p[i,Ny-1,k]
            pstar[i,Ny,k] = p[i,1,k]
            if i != Nx:
                ustar[i,0,k] = u[i,Ny-1,k]
                ustar[i,Ny,k] = u[i,1,k]
            if k != Nz:
                wstar[i,0,k] = w[i,Ny-1,k]
                wstar[i,Ny,k] = w[i,1,k]
    # No-slip at z-min and all-fixed at z-max
    for i in range(Nx+1):
        for j in range(Ny+1):
            wstar[i,j,0] = 0
            wstar[i,j,Nz-1] = 0
            pstar[i,j,0] = p[i,j,1]
            pstar[i,j,Nz] = p[i,j,Nz-1]
            if i != Nx:
                ustar[i,j,0] = -u[i,j,1]
                ustar[i,j,Nz] = 2 - u[i,j,Nz-1]
            if j != Ny:
                vstar[i,j,0] = - v[i,j,1]
                vstar[i,j,Nz] = - v[i,j,Nz-1]
    
    return ustar, vstar, wstar, pstar
    
@jit(nopython=True, fastmath=True)
def explicit(u,v,w):
    """
    Computes the explicit terms of the X, Y and Z-momentum equation based on the
    given U-velocity, V-velocity and W-velocity fields
    
    Parameters
    ==========
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    
    Returns
    =======
    Eu : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of explicit term of X-momentum equation at all the points
        of the U-velocity grid, values at boundary points being zero
    Ev : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of explicit term of Y-momentum equation at all the points
        of the V-velocity grid, values at boundary points being zero
    Ew : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of explicit term of Z-momentum equation at all the points
        of the W-velocity grid, values at boundary points being zero
    """
    Eu = np.zeros((Nx,Ny+1,Nz+1))
    Ev = np.zeros((Nx+1,Ny,Nz+1))
    Ew = np.zeros((Nx+1,Ny+1,Nz))
    
    for k in range(1,Nz):
        for j in range(1,Ny):
            for i in range(1,Nx-1):
                # U2x
                uavg2 = 0.5*(u[i+1,j,k]+u[i,j,k])
                uavg1 = 0.5*(u[i,j,k]+u[i-1,j,k])
                
                U2x = ((uavg2)**2-(uavg1)**2)/(xc[i+1]-xc[i])
                
                #UVy
                vavg2 = 0.5*(v[i,j,k]+v[i+1,j,k])
                vavg1 = 0.5*(v[i,j-1,k]+v[i+1,j-1,k])
                uavg2 = 0.5*(u[i,j,k]+u[i,j+1,k])
                uavg1 = 0.5*(u[i,j,k]+u[i,j-1,k])
                
                UVy = (uavg2*vavg2 - uavg1*vavg1)/(ye[j]-ye[j-1])
                
                #UWz
                wavg2 = 0.5*(w[i,j,k]+w[i+1,j,k])
                wavg1 = 0.5*(w[i,j,k-1]+w[i+1,j,k-1])
                uavg2 = 0.5*(u[i,j,k]+u[i,j,k+1])
                uavg1 = 0.5*(u[i,j,k]+u[i,j,k-1])
                
                UWz = (uavg2*wavg2 - uavg1*wavg1)/(ze[k]-ze[k-1])
                
                #Ux2
                Ux2 = ((u[i+1,j,k]-u[i,j,k])/(xe[i+1]-xe[i]) - \
                       (u[i,j,k]-u[i-1,j,k])/(xe[i]-xe[i-1]))/ \
                       (xc[i+1]-xc[i])
                
                #Uy2
                Uy2 = ((u[i,j+1,k]-u[i,j,k])/(yc[j+1]-yc[j]) - \
                       (u[i,j,k]-u[i,j-1,k])/(yc[j]-yc[j-1]))/ \
                       (ye[j]-ye[j-1])
                
                Eu[i,j,k] = U2x + UVy + UWz - (1/Re)*(Ux2 + Uy2)
    
    for i in range(1,Nx):
        for k in range(1,Nz):
            for j in range(1,Ny-1):
                # V2y
                vavg2 = 0.5*(v[i,j+1,k]+v[i,j,k])
                vavg1 = 0.5*(v[i,j,k]+v[i,j-1,k])
                
                V2y = ((vavg2)**2-(vavg1)**2)/(yc[j+1]-yc[j])
                
                #VUx
                uavg2 = 0.5*(u[i,j+1,k]+u[i,j,k])
                uavg1 = 0.5*(u[i-1,j,k]+u[i-1,j+1,k])
                vavg2 = 0.5*(v[i,j,k]+v[i+1,j,k])
                vavg1 = 0.5*(v[i,j,k]+v[i-1,j,k])
                
                VUx = (uavg2*vavg2 - uavg1*vavg1)/(xe[i]-xe[i-1])
                
                #VWz
                wavg2 = 0.5*(w[i,j,k]+w[i,j+1,k])
                wavg1 = 0.5*(w[i,j,k-1]+w[i,j+1,k-1])
                vavg2 = 0.5*(v[i,j,k]+v[i,j,k+1])
                vavg1 = 0.5*(v[i,j,k]+v[i,j,k-1])
                
                VWz = (vavg2*wavg2 - vavg1*wavg1)/(ze[k]-ze[k-1])
                
                #Vx2
                Vx2 = ((v[i+1,j,k]-v[i,j,k])/(xc[i+1]-xc[i]) - \
                       (v[i,j,k]-v[i-1,j,k])/(xc[i]-xc[i-1]))/ \
                       (xe[i]-xe[i-1])
                
                #Vy2
                Vy2 = ((v[i,j+1,k]-v[i,j,k])/(ye[j+1]-ye[j]) - \
                       (v[i,j,k]-v[i,j-1,k])/(ye[j]-ye[j-1]))/ \
                       (yc[j+1]-yc[j])
                
                Ev[i,j,k] = V2y + VUx + VWz - (1/Re)*(Vx2 + Vy2)
    
    for j in range(1,Ny):
        for i in range(1,Nx):
            for k in range(1,Nz-1):
                # W2z
                wavg2 = 0.5*(w[i,j,k+1]+w[i,j,k])
                wavg1 = 0.5*(w[i,j,k]+w[i,j,k-1])
                
                W2z = ((wavg2)**2-(wavg1)**2)/(zc[k+1]-zc[k])
                
                #WVy
                vavg2 = 0.5*(v[i,j,k]+v[i,j,k+1])
                vavg1 = 0.5*(v[i,j-1,k]+v[i,j-1,k+1])
                wavg2 = 0.5*(w[i,j,k]+w[i,j+1,k])
                wavg1 = 0.5*(w[i,j,k]+w[i,j-1,k])
                
                WVy = (wavg2*vavg2 - wavg1*vavg1)/(ye[j]-ye[j-1])
                
                #WUx
                wavg2 = 0.5*(w[i+1,j,k]+w[i,j,k])
                wavg1 = 0.5*(w[i,j,k]+w[i-1,j,k])
                uavg2 = 0.5*(u[i,j,k]+u[i,j,k+1])
                uavg1 = 0.5*(u[i-1,j,k]+u[i-1,j,k+1])
                
                WUx = (uavg2*wavg2 - uavg1*wavg1)/(xe[i]-xe[i-1])
                
                #Wx2
                Wx2 = ((w[i+1,j,k]-w[i,j,k])/(xc[i+1]-xc[i]) - \
                       (w[i,j,k]-w[i-1,j,k])/(xc[i]-xc[i-1]))/ \
                       (xe[i]-xe[i-1])
                
                #Wy2
                Wy2 = ((w[i,j+1,k]-w[i,j,k])/(yc[j+1]-yc[j]) - \
                       (w[i,j,k]-w[i,j-1,k])/(yc[j]-yc[j-1]))/ \
                       (ye[j]-ye[j-1])
                
                Ew[i,j,k] = W2z + WVy + WUx - (1/Re)*(Wx2 + Wy2)
    
    return Eu, Ev, Ew

@jit(nopython=True, fastmath=True)
def divergence(u,v,w,div):
    """
    Computes the divergence of velocity based on the given U-velocity,
    V-velocity and W-velocity fields
    
    Parameters
    ==========
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    div : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Empty matrix initialized with zeros
    
    Returns
    =======
    div : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
          Values of divergence of velocity field at all the points
          of the pressure grid, values at boundary points being zero
    """
    for k in range(1,Nz):
        for j in range(1,Ny):
            for i in range(1,Nx):    
                div[i,j,k] = (u[i,j,k]-u[i-1,j,k])/(xe[i]-xe[i-1]) + \
                             (v[i,j,k]-v[i,j-1,k])/(ye[j]-ye[j-1]) + \
                             (w[i,j,k]-w[i,j,k-1])/(ze[k]-ze[k-1])
    
    return div

@jit(nopython=True, fastmath=True)
def gauss_seidel_update(Arow,Acol,Adata,b,x,freq,diag):
    """
    Updates solution array x using Gauss-Seidel iterative technique
    for solving Ax = b where A is a sparse matrix
    
    Parameters
    ==========
    Arow  : numpy.ndarray of length nnz where nnz is the number of non-zero
            elements in A           
            Row indices of the non-zero elements of A in a row-major order
            default format: numpy.int32
    Acol  : numpy.ndarray of length nnz
            Column indices of the non-zero elements of A in a row-major order
            default format: numpy.int32
    Adata : numpy.ndarray of length nnz
            Values of the non-zero elements of A in a row-major order
            default format: numpy.float64
    b     : numpy.ndarray of length neq where neq is the number of equations
            Values of right hand side array
    x     : numpy.ndarray of length neq
            Values of initial solution array
    freq  : numpy.ndarray of length neq
            Number of non-zero elements in each row
    diag  : numpy.ndarray of length neq
            Indices of diagonal elements of A in Arow or Acol (diagonal elements 
            of A will have same position in Arow and Acol)
    
    Returns
    =======
    x     :  numpy.ndarray of length neq
            Values of updated solution array
    """
    k = len(freq)
    count = 0
    for i in range(k):
        x[i] = 1/Adata[diag[i]]*(b[i] - \
         x[Acol[count:count+freq[i]]].dot(Adata[count:count+freq[i]])\
         + x[Acol[diag[i]]]*(Adata[diag[i]]))
        count += freq[i]
    return x

def gseidel(A, b, x, maxiters=10000, tol=1e-6):
    """
    Computes the solution of Ax = b using Gauss-Seidel method
    
    Parameters
    ==========
    A : scipy.sparse.coo.coo_matrix
        Values of coefficient matrix
    b : numpy.ndarray of length neq where neq is the number of equations
        Values of right hand side array
    x : numpy.ndarray of length neq
        Values of initial solution array
    maxiters : numpy.int (optional)
        Total number of Gauss-Seidel Iterations
        default: 10000
    tol : numpy.float64 (optional)
        Tolerance of rms residual, below which solution is assumed to be
        converged
        default: 1e-6
    
    Returns
    =======
    x : numpy.ndarray of length neq
        Values of updated solution array
    """
    iteration = 0
    error = tol + 1
    neq = len(x)
    freq = np.bincount(A.row)
    diag = np.where(A.row == A.col)[0]
    while iteration < maxiters and error > tol:
        x = gauss_seidel_update(A.row, A.col, A.data,b,x,freq,diag)
        r = b - A.dot(x)
        norm_r = np.linalg.norm(r)
        error = np.sqrt(norm_r**2/neq)
        iteration += 1
#        While debugging, uncomment.
#        print('CPU Time = '+ str(timeit.default_timer()) + ', \
#        Iteration = '+ str(iteration) + ', |Residual| = '+ str(error) + '\n')
    return x

def generate_A(ijklevels):
    """
    Generates the coefficient matrices A for all the levels of multigrid solver
    
    Parameters
    ==========
    ijklevels : numpy.ndarray of shape (ndof,nl)
                where, ndof is the number of degrees of freedom
                and, nl is the number of levels
        Values of number of grid nodes at each level for each degree of freedom
    
    Returns
    =======
    A : list of length nl containing elements of format scipy.sparse.coo.coo_matrix
        Sparse coefficient matrices for all the levels of multigrid solver
    """
    numLevels = ijklevels.shape[1]
    A = []    
    for i in range(numLevels):
        nx = ijklevels[0][i]-2
        ny = ijklevels[1][i]-2
        nz = ijklevels[2][i]-2
        diagonalsx = [np.ones(nx-1), -2*np.ones(nx), np.ones(nx-1)]
        diagonalsy = [np.ones(ny-1), -2*np.ones(ny), np.ones(ny-1)]
        diagonalsz = [np.ones(nz-1), -2*np.ones(nz), np.ones(nz-1)]
        Ax = sp.diags(diagonalsx,[-1,0,1])
        Ay = sp.diags(diagonalsy,[-1,0,1])
        Az = sp.diags(diagonalsz,[-1,0,1])
        A.append(sp.kronsum(sp.kronsum(Ax,Ay),Az,format='coo'))
    return A

def generate_b(ijklevels, phi, f):
    """
    Generates the right hand side matrices b of all the levels of multigrid solver
    
    Parameters
    ==========
    ijklevels : numpy.ndarray of shape (ndof,nl)
                where, ndof is the number of degrees of freedom
                and, nl is the number of levels
        Values of number of grid nodes at each level for each degree of freedom
    phi : numpy.ndarray of shape (n1,n2,n3)
          where, n1 is the number of grid points on the finest grid in x-direction,
                 n2 is the number of grid points on the finest grid in y-direction,
            and, n3 is the number of grid points on the finest grid in z-direction
        Array of unknowns, with only the boundary values being non-zero and
        all the interior values being zero
    f : numpy.ndarray of shape (n1,n2,n3)
        Values of the right hand side of the discretized equation at all the points,
        boundary values being zero.
    
    Returns
    =======
    b : list of length nl
        Right hand side arrays for all the levels of multigrid solver
    """
    numLevels = ijklevels.shape[1]
    b = []
    for ilevel in range(numLevels):
        Nx = ijklevels[0][ilevel]
        Ny = ijklevels[1][ilevel]
        Nz = ijklevels[2][ilevel]
        nx = Nx-2
        ny = Ny-2
        nz = Nz-2
        neq = nx*ny*nz
        g = np.empty(neq)
        b_level = np.empty(neq)
        d = 2**ilevel
        xindices = np.arange(0,ijklevels[0][0],d)
        yindices = np.arange(0,ijklevels[1][0],d)
        zindices = np.arange(0,ijklevels[2][0],d)
        for i in xindices[1:-1]:
            for j in yindices[1:-1]: 
                g[(int(i/d)-1)+nx*(int(j/d)-1):neq:nx*ny] = - (phi[i-d,j,zindices[1:-1]] + \
                  phi[i+d,j,zindices[1:-1]] + phi[i,j+d,zindices[1:-1]] + phi[i,j-d,zindices[1:-1]] \
                  + phi[i,j,zindices[0:-2]] + phi[i,j,zindices[2:]])
                b_level[(int(i/d)-1)+nx*(int(j/d)-1):neq:nx*ny] = hx**2*f[i,j,zindices[1:-1]] \
                + g[(int(i/d)-1)+nx*(int(j/d)-1):neq:nx*ny]
        b.append(b_level)
    return b

def restrict(xfe, nfx, nfy, nfz, memory, notInMemory = True):
    """
    Applies the restriction operation on solution array on a fine grid to obtain
    the solution array on a coarse grid
    
    Parameters
    ==========
    xfe : numpy.ndarray of length nfeq
          where, nfeq is the number of equations in the fine grid
        Values of solution array on the fine grid
    nfx : numpy.int
        Number of grid points on the fine grid in x-direction
    nfy : numpy.int
        Number of grid points on the fine grid in y-direction
    nfz : numpy.int
        Number of grid points on the fine grid in z-direction
    memory : list
        Cache of restriction matrices along with its associated grid size information.
        If the restriction matrix of the given fine grid is already present in the memory,
        there is no need to compute it. This saves a lot of computation time.
        format: [[(nfx0,nfy0,nfz0),(nfx1,nfy1,nfz1),...],[R0,R1,...]]
            where nfx0,nfx1,...: Number of grid points in x-direction of the fine grids
                                 whose restriction matrices are stored,
                                 format: numpy.int
                  nfy0,nfy1,...: Number of grid points in y-direction of the fine grids
                                 whose restriction matrices are stored,
                                 format: numpy.int
                  nfz0,nfz1,...: Number of grid points in z-direction of the fine grids
                                 whose restriction matrices are stored,
                                 format: numpy.int
                  R0,R1,...: Restriction matrices stored
                             format: scipy.sparse.coo.coo_matrix
    notInMemory : bool (optional)
        Tells whether the restriction matrix is already present in the memory
        for the given fine grid parameters
    
    Returns
    =======
    xce : numpy.ndarray of length nceq
          where, nceq is the number of equations in the coarse grid
        Values of computed solution array on the coarse grid
    ncx : numpy.int
        Number of grid points on the coarse grid in x-direction
    ncy : numpy.int
        Number of grid points on the coarse grid in y-direction
    ncz : numpy.int
        Number of grid points on the coarse grid in z-direction
    """
    ncx = int((nfx+1)/2) - 1
    ncy = int((nfy+1)/2) - 1
    ncz = int((nfz+1)/2) - 1
    for i, value in enumerate(memory[0]):
        if(value == (nfx,nfy,nfz)):
            R3D = memory[1][i].copy()
            notInMemory = False
            break
    if notInMemory:
        block = np.array([1, 2, 1])
        Rx = sp.lil_matrix((ncx,nfx))
        Ry = sp.lil_matrix((ncy,nfy))
        Rz = sp.lil_matrix((ncz,nfz))
        for i in range(ncx):
            Rx[i, 2*i] = 0.25*block[0]
            Rx[i, 2*i+1] = 0.25*block[1]
            Rx[i, 2*i+2] = 0.25*block[2]
        for j in range(ncy):
            Ry[j, 2*j] = 0.25*block[0]
            Ry[j, 2*j+1] = 0.25*block[1]
            Ry[j, 2*j+2] = 0.25*block[2]
        for k in range(ncz):
            Rz[k, 2*k] = 0.25*block[0]
            Rz[k, 2*k+1] = 0.25*block[1]
            Rz[k, 2*k+2] = 0.25*block[2]
        R3D = sp.kron(Rz,sp.kron(Ry,Rx))
        memory[0].append((nfx,nfy,nfz))
        memory[1].append(R3D)
    xce = R3D.dot(xfe)
    
    return xce, ncx, ncy, ncz

def prolong(xce, ncx, ncy, ncz, memory, notInMemory = True):
    """
    Applies the prolongation operation on solution array on a coarse grid to obtain
    the solution array on a fine grid
    
    Parameters
    ==========
    xce : numpy.ndarray of length nceq
          where, nceq is the number of equations in the coarse grid
        Values of solution array on the coarse grid
    ncx : numpy.int
        Number of grid points on the coarse grid in x-direction
    ncy : numpy.int
        Number of grid points on the coarse grid in y-direction
    ncz : numpy.int
        Number of grid points on the coarse grid in z-direction
    memory : list
        Cache of prolongation matrices along with its associated grid size information.
        If the prolongation matrix of the given coarse grid is already present in the memory,
        there is no need to compute it. This saves a lot of computation time.
        format: [[(ncx0,ncy0,ncz0),(ncx1,ncy1,ncz1),...],[P0,P1,...]]
            where ncx0,ncx1,...: Number of grid points in x-direction of the coarse grids
                                 whose prolongation matrices are stored,
                                 format: numpy.int
                  ncy0,ncy1,...: Number of grid points in y-direction of the coarse grids
                                 whose prolongation matrices are stored,
                                 format: numpy.int
                  ncz0,ncz1,...: Number of grid points in z-direction of the coarse grids
                                 whose prolongation matrices are stored,
                                 format: numpy.int
                  P0,P1,...: Prolongation matrices stored
                             format: scipy.sparse.coo.coo_matrix
    notInMemory : bool (optional)
        Tells whether the prolongation matrix is already present in the memory
        for the given coarse grid parameters
    
    Returns
    =======
    xfe : numpy.ndarray of length nfeq
          where, nfeq is the number of equations in the fine grid
        Values of computed solution array on the fine grid
    nfx : numpy.int
        Number of grid points on the fine grid in x-direction
    nfy : numpy.int
        Number of grid points on the fine grid in y-direction
    nfz : numpy.int
        Number of grid points on the fine grid in z-direction    
    """
    nfx = (ncx+1)*2 - 1
    nfy = (ncy+1)*2 - 1
    nfz = (ncz+1)*2 - 1
    for i, value in enumerate(memory[0]):
        if(value == (ncx,ncy,ncz)):
            I3D = memory[1][i].copy()
            notInMemory = False
            break
    if notInMemory:
        block = np.array([1, 2, 1])
        Ix = sp.lil_matrix((nfx,ncx))
        Iy = sp.lil_matrix((nfy,ncy))
        Iz = sp.lil_matrix((nfz,ncz))
        for i in range(ncx):
            Ix[2*i, i] = 0.5*block[0]
            Ix[2*i+1, i] = 0.5*block[1]
            Ix[2*i+2, i] = 0.5*block[2]
        for j in range(ncy):
            Iy[2*j, j] = 0.5*block[0]
            Iy[2*j+1, j] = 0.5*block[1]
            Iy[2*j+2, j] = 0.5*block[2]
        for k in range(ncz):
            Iz[2*k, k] = 0.5*block[0]
            Iz[2*k+1, k] = 0.5*block[1]
            Iz[2*k+2, k] = 0.5*block[2]
        I3D = sp.kron(Iz,sp.kron(Iy,Ix))
        memory[0].append((ncx,ncy,ncz))
        memory[1].append(I3D)
    xfe = I3D.dot(xce)
    
    return xfe, nfx, nfy, nfz

def mg_update(A,b,x,level):
    """
    Updates the solution array x after one W-cycle.
    
    Parameters
    ==========
    A : list of length nl containing elements of format scipy.sparse.coo.coo_matrix
        where nl is the number of levels
        Sparse coefficient matrices for all the levels of multigrid solver
    b : numpy.ndarray of length neq where neq is the number of equations in that level
        Values of right hand side array
    x : numpy.ndarray of length neq
        Values of initial solution array
    level: numpy.int
        Current level
    
    Returns
    =======
    x : numpy.ndarray of length neq
        Values of updated solution array
    
    """
    #Pre-Smoothing
    if level == numLevels-2:
        x = gseidel(A[level], b, x, maxiters=4)
    elif level == numLevels-3:
        x = gseidel(A[level], b, x, maxiters=2)
    elif level == numLevels-4:
        x = gseidel(A[level], b, x, maxiters=1)
    else:
        x = gseidel(A[level], b, x, maxiters=1)
    #compute residual
    rf = b - A[level].dot(x)
    #Restriction
    nfx = ijkLevels[0][level] - 2
    nfy = ijkLevels[1][level] - 2
    nfz = ijkLevels[2][level] - 2
    rc, ncx, ncy, ncz = restrict(rf, nfx, nfy, nfz, restrict_memory)
    eps = np.zeros(len(rc))
    if level == numLevels-2:
        eps = gseidel(A[level+1], rc, eps)
    else:
        eps = mg_update(A, rc, eps, level+1)
    #Prolongation
    epsf, nfx, nfy, nfz = prolong(eps, ncx, ncy, ncz, prolong_memory)
    x = x + epsf.copy()
    #compute residual
    rf = b - A[level].dot(x)
    #Restriction
    rc, ncx, ncy, ncz = restrict(rf, nfx, nfy, nfz, restrict_memory)
    if level == numLevels-2:
        eps = gseidel(A[level+1], rc, eps)
    else:
        eps = mg_update(A, rc, eps, level+1)
    #Prolongation
    epsf, nfx, nfy, nfz = prolong(eps, ncx, ncy, ncz, prolong_memory)
    x = x + epsf.copy()
    #Post-smoothing
    if level == numLevels-2:
        x = gseidel(A[level], b, x, maxiters=4)
    elif level == numLevels-3:
        x = gseidel(A[level], b, x, maxiters=2)
    elif level == numLevels-4:
        x = gseidel(A[level], b, x, maxiters=1)  
    else:
        x = gseidel(A[level], b, x, maxiters=1)
    
    return x


def multigrid(A,b,xe,maxiter=10000,tol = 1e-6):
    """
    Computes the solution of Ax = b using Multigrid method
    
    Parameters
    ==========
    A : list of length nl containing elements of format scipy.sparse.coo.coo_matrix
        where nl is the number of levels
        Sparse coefficient matrices for all the levels of multigrid solver
    b : list of length nl
        Right hand side arrays for all the levels of multigrid solver
    xe : numpy.ndarray of length neq
        Values of initial solution array
    maxiters : numpy.int (optional)
        Total number of Multigrid Iterations
        default: 10000
    tol : numpy.float64 (optional)
        Tolerance of rms residual, below which solution is assumed to be
        converged
        default: 1e-6
    
    Returns
    =======
    x : numpy.ndarray of length neq
        Values of updated solution array
    """
    iteration = 0
    r = b[0] - A[0].dot(xe)
    neq = len(r)
    norm_r = np.linalg.norm(r)
    error = np.sqrt(norm_r**2/neq)
    time = timeit.default_timer()
    #f.write(str(time) + '\t' + str(iteration) + '\t' + str(norm_r) + '\n')
#    print('CPU Time = '+ str(time) + ', V-cycle Iteration = ' + str(iteration) + ', |Residual| = ' + str(error))

    while iteration < maxiter and error > tol:
#        print(timeit.default_timer())
        xe = mg_update(A, b[0], xe, 0)
#        print(timeit.default_timer())
        r = b[0] - A[0].dot(xe)
        norm_r = np.linalg.norm(r)
        error = np.sqrt(norm_r**2/neq)
        iteration += 1
#        time = timeit.default_timer()
        #f.write(str(time) + '\t' + str(iteration) + '\t' + str(norm_r) + '\n')
#        print('CPU Time = '+ str(time) + ', V-cycle Iteration = ' + str(iteration) + ', |Residual| = ' + str(error))
    return xe

@jit(nopython=True)
def initguess(d,x):
    """
    Gets a decent initial guess of the solution array x by solving Kx=d where K
    is the following tridiagonal matrix.
    K = array([[-6.,  1.,  0., ...,  0.,  0.,  0.],
              [ 1., -6.,  1., ...,  0.,  0.,  0.],
              [ 0.,  1., -6., ...,  0.,  0.,  0.],
              ...,
              [ 0.,  0.,  0., ..., -6.,  1.,  0.],
              [ 0.,  0.,  0., ...,  1., -6.,  1.],
              [ 0.,  0.,  0., ...,  0.,  1., -6.]]),
    where the shape of K is (neq,neq), neq is the number of equations. 
    
    Parameters
    ==========
    d : numpy.ndarray of length neq
        Values of the right hand side array
    x : numpy.ndarray of length neq
        Uninitialized solution array
    
    Returns
    =======
    numpy.ndarray of length neq
    Updated solution array
    """
    N = len(x)
    a = np.ones(N-1)
    b = (-6)*np.ones(N)
    c = np.ones(N-1)
    return thomas(a,b,c,d,x)

def solve_p2(rhs,p):
    """
    Solves the Pressure Poisson equation using Multigrid method
    
    Parameters
    ==========
    rhs : numpy.ndarray of shape (n1,n2,n3)
          where, n1, n2 and n3 are the number of grid points in x, y and z-directions
          in pressure grid
        Values of the right hand side of the discretized equation at all the points,
        boundary values being zero.
    p : numpy.ndarray of shape (n1,n2,n3)
        Array of pressure values, with only the boundary values being non-zero and
        all the interior values being zero
    
    Returns
    =======
    p : numpy.ndarray of shape (n1,n2,n3)
        Values of computed pressure at all the points on the pressure grid
    """
    b = generate_b(ijkLevels, p, rhs) 
    nx = Nx-1
    ny = Ny-1
    nz = Nz-1
    neq = nx*ny*nz
#    p1d = np.zeros(neq)
#    p1d= np.ones(neq)
#    p1d = -(1/6)*b[0].copy()
    p1d = np.empty(neq)
    p1d = initguess(b[0].copy(),p1d)
    p1d = multigrid(A,b,p1d)
    p[1:nx+1,1:ny+1,1:nz+1] = p1d.reshape((nx,ny,nz),order='F')
    return p
               
def solve_p(rhs,p):
    """
    Solves the Pressure Poisson equation using Gauss-Seidel method
    
    Parameters
    ==========
    rhs : numpy.ndarray of shape (n1,n2,n3)
          where, n1, n2 and n3 are the number of grid points in x, y and z-directions
          in pressure grid
        Values of the right hand side of the discretized equation at all the points,
        boundary values being zero.
    p : numpy.ndarray of shape (n1,n2,n3)
        Array of pressure values, with only the boundary values being non-zero and
        all the interior values being zero
    
    Returns
    =======
    p : numpy.ndarray of shape (n1,n2,n3)
        Values of computed pressure at all the points on the pressure grid
    """
    nx = Nx-1
    ny = Ny-1
    nz = Nz-1
    neq = nx*ny*nz
    g = np.empty(neq)
    b = np.empty(neq)
    diagonalsx = [np.ones(nx-1), -2*np.ones(nx), np.ones(nx-1)]
    diagonalsy = [np.ones(ny-1), -2*np.ones(ny), np.ones(ny-1)]
    diagonalsz = [np.ones(nz-1), -2*np.ones(nz), np.ones(nz-1)]
    Ax = sp.diags(diagonalsx,[-1,0,1])
    Ay = sp.diags(diagonalsy,[-1,0,1])
    Az = sp.diags(diagonalsz,[-1,0,1])
    A = sp.kronsum(sp.kronsum(Ax,Ay),Az,format='coo')
    for i in range(1,nx+1):
        for j in range(1,ny+1): 
            g[(i-1)+nx*(j-1):neq:nx*ny] = - (p[i-1,j,1:nz+1] + \
              p[i+1,j,1:nz+1] + p[i,j,0:nz] + p[i,j,2:nz+2] + \
              p[i,j-1,1:nz+1]+ p[i,j+1,1:nz+1])
            b[(i-1)+nx*(j-1):neq:nx*ny] = hx**2*rhs[i,j,1:nz+1] \
            + g[(i-1)+nx*(j-1):neq:nx*ny]
    p1d = np.ones(neq)
    p1d = gseidel(A,b,p1d)
    p[1:nx+1,1:ny+1,1:nz+1] = p1d.reshape((nx,ny,nz),order='F')             
    return p

@jit(nopython=True, fastmath=True)
def delp(p):
    """
    Returns all three directional derivatives of pressure based on the 
    pressure field given
    
    Parameters
    ==========
    p : numpy.ndarray of shape (n1,n2,n3)
          where, n1, n2 and n3 are the number of grid points in x, y and z-directions
          in pressure grid
        Values of pressure at all the points on the pressure grid
    
    Returns
    =======
    Px : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
         where, Nx, Ny, Nz are the number of grid points in x, y and z direction for
         the nodal grid
        Values of dp/dx at all points on the U-velocity grid
    Py : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of dp/dy at all points on the V-velocity grid
    Pz : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of dp/dz at all points on the W-velocity grid
    """
    Px = np.zeros((Nx,Ny+1,Nz+1))
    Py = np.zeros((Nx+1,Ny,Nz+1))
    Pz = np.zeros((Nx+1,Ny+1,Nz))
    for k in range(1,Nz):
        for j in range(1,Ny):
            for i in range(1,Nx-1):
                Px[i,j,k] = (p[i+1,j,k] - p[i,j,k])/(xc[i+1] - xc[i])
    for i in range(1,Nx):
        for k in range(1,Nz):
            for j in range(1,Ny-1):
                Py[i,j,k] = (p[i,j+1,k] - p[i,j,k])/(yc[j+1] - yc[j])
    for j in range(1,Ny):
        for i in range(1,Nx):
            for k in range(1,Nz-1):
                Pz[i,j,k] = (p[i,j,k+1] - p[i,j,k])/(zc[k+1] - zc[k])
    return Px, Py, Pz

def rk(c1, c2, c3, delt, u, v, w, p, qu, qv, qw, Eu, Ev, Ew):
    """
    Updates u, v, w, p for an RK substep in RK3 Williamson method.
    
    Parameters
    ==========
    c1, c2, c3 : numpy.float64
        For RK3 Williamson method, following are the values for each RK
        substep:
            Substep 1: c1 = 0, c2 = 1/3, c3 = 1/3
            Substep 2: c1 = -5/9, c2 = 15/16, c3 = 5/12
            Substep 3: c1 = -153/128, c2 = 8/15, c3 = 1/4
    delt : numpy.float64
        Timestep
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid 
    Eu : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of explicit term of X-momentum equation at all the points
        of the U-velocity grid, values at boundary points being zero
    Ev : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of explicit term of Y-momentum equation at all the points
        of the V-velocity grid, values at boundary points being zero
    Ew : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of explicit term of Z-momentum equation at all the points
        of the W-velocity grid, values at boundary points being zero        
    qu : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Substep1: qu1 = delt*Eu1
        Substep2: qu2 = c1*qu1 + delt*Eu2
        Substep3: qu3 = c1*qu2 + delt*Eu3
        where, c1 corresponds to its value at each substep
    qv : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
    qw : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
    
    Returns
    =======
    ustar : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of updated U-velocity at all the points of the U-velocity grid
    vstar : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of updated V-velocity at all the points of the V-velocity grid
    wstar : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of updated W-velocity at all the points of the W-velocity grid
    pstar : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of updated pressure at all the points of the pressure grid
    qu : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Updated values of qu
    qv : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Updated values of qv
    qw : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Updated values of qw
    """
    deltrk = c3*delt
    qu = c1*qu + delt*Eu
    qv = c1*qv + delt*Ev
    qw = c1*qw + delt*Ew
    rhsu = u - c2*qu
    rhsv = v - c2*qv
    rhsw = w - c2*qw
    ustar = np.zeros((Nx,Ny+1,Nz+1))
    vstar = np.zeros((Nx+1,Ny,Nz+1))
    wstar = np.zeros((Nx+1,Ny+1,Nz))
    pstar = np.zeros((Nx+1,Ny+1,Nz+1))
    ustar, vstar, wstar, pstar = equatebc(ustar, vstar, wstar, pstar, u, v, w, p)
    for i in range(1,Nx-1):
        for j in range(1,Ny):
            rtemp = rhsu[i,j,1:Nz].copy()
            rtemp[0] += deltrk/(Re*hz**2)*ustar[i,j,0]
            rtemp[-1] += deltrk/(Re*hz**2)*ustar[i,j,Nz]
            lower = -deltrk/(Re*hz**2)*np.ones(Nz-2)
            upper = -deltrk/(Re*hz**2)*np.ones(Nz-2)
            diag = (1+2*deltrk/(Re*hz**2))*np.ones(Nz-1)
            utemp = ustar[i,j,1:Nz].copy()
            utemp = thomas(lower, diag, upper, rtemp, utemp)
            ustar[i,j,1:Nz] = utemp.copy()
    for i in range(1,Nx):
        for j in range(1,Ny-1):
            rtemp = rhsv[i,j,1:Nz].copy()
            rtemp[0] += deltrk/(Re*hz**2)*vstar[i,j,0]
            rtemp[-1] += deltrk/(Re*hz**2)*vstar[i,j,-1]
            lower = -deltrk/(Re*hz**2)*np.ones(Nz-2)
            upper = -deltrk/(Re*hz**2)*np.ones(Nz-2)
            diag = (1+2*deltrk/(Re*hz**2))*np.ones(Nz-1)
            vtemp = vstar[i,j,1:Nz].copy()
            vtemp = thomas(lower, diag, upper, rtemp, vtemp)
            vstar[i,j,1:Nz] = vtemp.copy()
    for i in range(1,Nx):
        for j in range(1,Ny):
            rtemp = rhsw[i,j,1:Nz-1].copy()
            rtemp[0] += deltrk/(Re*hz**2)*wstar[i,j,0]
            rtemp[-1] += deltrk/(Re*hz**2)*wstar[i,j,-1]
            lower = -deltrk/(Re*hz**2)*np.ones(Nz-3)
            upper = -deltrk/(Re*hz**2)*np.ones(Nz-3)
            diag = (1+2*deltrk/(Re*hz**2))*np.ones(Nz-2)
            wtemp = wstar[i,j,1:Nz-1].copy()
            wtemp = thomas(lower, diag, upper, rtemp, wtemp)
            wstar[i,j,1:Nz-1] = wtemp.copy()
    div = np.zeros((Nx+1,Ny+1,Nz+1))
    div = divergence(ustar,vstar,wstar,div)
    rhsp = (1/deltrk)*div
    pstar = solve_p2(rhsp, pstar)
    px, py, pz = delp(pstar)
    ustar = ustar - deltrk*px
    vstar = vstar - deltrk*py
    wstar = wstar - deltrk*pz
    return ustar, vstar, wstar, pstar, qu, qv, qw

def savedata(x,y,z,u,v,w,p,imax,jmax,kmax,iters):
    """
    Saves U-velocity, V-velocity, W-velocity and pressure field data in a
    Tecplot readable ASCII file
    
    Parameters
    ==========
    x, y, z : numpy.ndarray of shape (Nx,Ny,Nz)
        coordinate data
    u, v, w, p : numpy.ndarray of shape (Nx,Ny,Nz)
        field data
    imax, jmax, kmax : numpy.int
        Number of grid points in x, y and z-direction
    iters : numpy.int
        Number of iterations needed to generate this data
    
    Returns
    =======
    Creates a Tecplot readable ASCII file in the desired location 
    """
    file = open("C:/Users/debar/Desktop/python_tecplot/ldc33_"+str(iters)+".dat",'w')
    file.write("TITLE = \"Lid Driven Cavity Data\" \n")
    file.write("VARIABLES = \"X\" \"Y\" \"Z\" \"U\" \"V\" \"W\" \"P\" \n")
    file.write("ZONE \n")
    file.write("ZONETYPE = ORDERED, I = " + str(imax) + \
               ", J = " + str(jmax) + ", K = " + str(kmax) + "\n")
    file.write("DATAPACKING = POINT \n")
    for k in range(kmax):
        for j in range(jmax):
            for i in range(imax):
                file.write(str(x[i,j,k]) + '\t' + str(y[i,j,k]) + '\t' + \
                           str(z[i,j,k]) + '\t' + str(u[i,j,k]) + '\t' + \
                           str(v[i,j,k]) + '\t' + str(w[i,j,k]) + '\t' + \
                           str(p[i,j,k]) + '\n')
    file.close()

@jit(nopython=True)
def compute_residuals(Eu, Ev, Ew, u, v, w, p, u_old, v_old, w_old, delt):
    """
    Computes residuals for continuity, X-momentum, Y-momentum, Z-momentum
    equations at each point
    
    Parameters
    ==========
    Eu : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of explicit term of X-momentum equation at all the points
        of the U-velocity grid, values at boundary points being zero
    Ev : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of explicit term of Y-momentum equation at all the points
        of the V-velocity grid, values at boundary points being zero
    Ew : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of explicit term of Z-momentum equation at all the points
        of the W-velocity grid, values at boundary points being zero  
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid 
    u_old : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity of previous timestep at all the points
        of the U-velocity grid
    v_old : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity of previous timestep at all the points
        of the V-velocity grid
    w_old : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity of previous timestep at all the points\
        of the W-velocity grid
    delt : numpy.float64
        Timestep
    
    Returns
    =======
    div : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
          Values of divergence of velocity field at all the points
          of the pressure grid, values at boundary points being zero
    residualU : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of residuals of X-momentum equation at all the points
        of the U-velocity grid
    residualV : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of residuals of Y-momentum equation at all the points
        of the V-velocity grid
    residualW : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of residuals of Z-momentum equation at all the points
        of the W-velocity grid
    """
    Iu = np.zeros((Nx,Ny+1,Nz+1))
    Iv = np.zeros((Nx+1,Ny,Nz+1))
    Iw = np.zeros((Nx+1,Ny+1,Nz))
    div = np.zeros((Nx+1,Ny+1,Nz+1))
    
    for k in range(1,Nz):
        for j in range(1,Ny):
            for i in range(1,Nx-1):
                Iu[i,j,k] = (1/Re)*((u[i,j,k+1]-u[i,j,k])/(zc[k+1]-zc[k]) - \
                            (u[i,j,k]-u[i,j,k-1])/(zc[k]-zc[k-1]))/ \
                            (ze[k]-ze[k-1])
    
    for i in range(1,Nx):
        for k in range(1,Nz):
            for j in range(1,Ny-1):
                Iv[i,j,k] =(1/Re)*((v[i,j,k+1]-v[i,j,k])/(zc[k+1]-zc[k]) - \
                           (v[i,j,k]-v[i,j,k-1])/(zc[k]-zc[k-1]))/ \
                           (ze[k]-ze[k-1])
    
    for j in range(1,Ny):
        for i in range(1,Nx):
            for k in range(1,Nz-1):
                Iw[i,j,k] = (1/Re)*((w[i,j,k+1]-w[i,j,k])/(ze[k+1]-ze[k]) - \
                       (w[i,j,k]-w[i,j,k-1])/(ze[k]-ze[k-1]))/ \
                       (zc[k+1]-zc[k])
    
    px, py, pz = delp(p)
    
    div = divergence(u,v,w,div)
    
    residualU = (u-u_old)/delt + Eu + px - Iu
    residualV = (v-v_old)/delt + Ev + py - Iv
    residualW = (w-w_old)/delt + Ew + pz - Iw
    
    return div, residualU, residualV, residualW


def residual(Eu, Ev, Ew, u, v, w, p, u_old, v_old, w_old, delt):
    """
    Computes a combined residual for checking convergence
    
    Parameters
    ==========
    Eu : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of explicit term of X-momentum equation at all the points
        of the U-velocity grid, values at boundary points being zero
    Ev : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of explicit term of Y-momentum equation at all the points
        of the V-velocity grid, values at boundary points being zero
    Ew : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of explicit term of Z-momentum equation at all the points
        of the W-velocity grid, values at boundary points being zero  
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid 
    u_old : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity of previous timestep at all the points
        of the U-velocity grid
    v_old : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity of previous timestep at all the points
        of the V-velocity grid
    w_old : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity of previous timestep at all the points\
        of the W-velocity grid
    delt : numpy.float64
        Timestep
    
    Returns
    =======
    res : numpy.float64
        Combined residual defined by the arithmetic mean of the 
        rms residuals of continuity, X-momentum, Y-momentum, Z-momentum
        equations
    """
    div, residualU, residualV, residualW = compute_residuals(Eu, Ev, \
                             Ew, u, v, w, p, u_old, v_old, w_old, delt)
    
    normU = np.linalg.norm(residualU)
    normV = np.linalg.norm(residualV)
    normW = np.linalg.norm(residualW)
    normdiv = np.linalg.norm(div)
    
#    print(np.linalg.norm(Eu),np.linalg.norm(px),np.linalg.norm(Iu))
#    print(np.linalg.norm(Ev),np.linalg.norm(py),np.linalg.norm(Iv))
#    print(np.linalg.norm(Ew),np.linalg.norm(pz),np.linalg.norm(Iw))
#    print(normdiv)
    
    res = np.sqrt(normU**2+normV**2+normW**2+normdiv**2)

#    res = 0.25*(np.sqrt(normU**2/((Nx-1)*Ny*Nz)) + \
#          np.sqrt(normV**2/(Nx*(Ny-1)*Nz)) + \
#          np.sqrt(normW**2/(Nx*Ny*(Nz-1))) + \
#          np.sqrt(normdiv**2/(Nx*Ny*Nz)))
    
    return res

@jit(nopython=True)
def convertToNodalData(u,v,w,p):
    """
    Converts staggered grid data to collocated grid data.
    
    Parameters
    ==========
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid
        
    Returns
    =======
    un, vn, wn, pn: numpy.ndarray of shape (Nx,Ny,Nz)
        Values of U-velocity, V-velocity, W-velocity and pressure at all
        points of the collocated grid
    """
    Nx = u.shape[0]
    Ny = v.shape[1]
    Nz = w.shape[2]
    
    un = np.empty((Nx,Ny,Nz))
    vn = np.empty((Nx,Ny,Nz))
    wn = np.empty((Nx,Ny,Nz))
    pn = np.empty((Nx,Ny,Nz))
    
    utemp = np.empty((Nx,Ny,Nz+1))
    vtemp = np.empty((Nx+1,Ny,Nz))
    wtemp = np.empty((Nx,Ny+1,Nz))
    
    for k in range(Nz+1):
        for i in range(Nx):
            utemp[i,:,k] = 0.5*(u[i,:-1,k]+u[i,1:,k])
    
    for j in range(Ny):
        for i in range(Nx):
            un[i,j,:] = 0.5*(utemp[i,j,:-1]+utemp[i,j,1:])

    for i in range(Nx+1):
        for j in range(Ny):
            vtemp[i,j,:] = 0.5*(v[i,j,:-1]+v[i,j,1:])
    
    for j in range(Ny):
        for k in range(Nz):
            vn[:,j,k] = 0.5*(vtemp[1:,j,k]+vtemp[:-1,j,k])

    for j in range(Ny+1):
        for k in range(Nz):
            wtemp[:,j,k] = 0.5*(w[:-1,j,k]+w[1:,j,k])
    
    for k in range(Nz):
        for i in range(Nx):
            wn[i,:,k] = 0.5*(wtemp[i,1:,k]+wtemp[i,:-1,k])
    
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                pn[i,j,k] = (1/8)*(p[i,j,k] + p[i+1,j,k] +\
                 p[i,j+1,k] + p[i,j,k+1] + p[i+1,j+1,k] + p[i,j+1,k+1] +\
                 p[i+1,j,k+1] + p[i+1,j+1,k+1])
            
    return un, vn, wn, pn

def saverestartdata(u,v,w,p,iters,time):
    """
    Generates a restart datafile in order to restart from that iteration if
    necessary.
    
    Parameters
    ==========
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid
    iters : numpy.int
        Number of iterations at which the restart data is saved
    time : numpy.float
        Time at which the restart data is saved
    
    Returns
    =======
    Creates an ASCII file with the following format:
        iters
        time
        u
        v
        w
        p
    
    """
    file = open("C:/Users/debar/Desktop/python_tecplot/ldc33res_"+str(iters)+".dat",'w')
    file.write(str(iters) + '\n')
    file.write(str(time) + '\n')
    
    for k in range(Nz+1):
        for j in range(Ny+1):
            for i in range(Nx):
                file.write(str(u[i,j,k])+ '\n')
    
    for k in range(Nz+1):
        for j in range(Ny):
            for i in range(Nx+1):
                file.write(str(v[i,j,k])+ '\n')
    
    for k in range(Nz):
        for j in range(Ny+1):
            for i in range(Nx+1):
                file.write(str(w[i,j,k])+ '\n')
    
    for k in range(Nz+1):
        for j in range(Ny+1):
            for i in range(Nx+1):
                file.write(str(p[i,j,k])+ '\n')

def readdata(file):
    """
    Reads data from a restart datafile
    
    Parameters
    ==========
    file : _io.TextIOWrapper
        The restart file text IO wrapper
    
    Returns
    =======
    u : numpy.ndarray of shape (Nx,Ny+1,Nz+1)
        Values of U-velocity at all the points of the U-velocity grid
    v : numpy.ndarray of shape (Nx+1,Ny,Nz+1)
        Values of V-velocity at all the points of the V-velocity grid
    w : numpy.ndarray of shape (Nx+1,Ny+1,Nz)
        Values of W-velocity at all the points of the W-velocity grid
    p : numpy.ndarray of shape (Nx+1,Ny+1,Nz+1)
        Values of pressure at all the points of the pressure grid
    iters : numpy.int
        Number of iterations at which the restart data is saved
    time : numpy.float
        Time at which the restart data is saved
    """
    iters = float(file.readline())
    time = float(file.readline())
    a = np.loadtxt(file, dtype='float64')
    print(a.shape)
    uend = Nx*(Ny+1)*(Nz+1)
    vend = uend + Ny*(Nx+1)*(Nz+1)
    wend = vend + Nz*(Nx+1)*(Ny+1)
    u = a[:uend].reshape((Nx,Ny+1,Nz+1),order='F')
    v = a[uend:vend].reshape((Nx+1,Ny,Nz+1),order='F')
    w = a[vend:wend].reshape((Nx+1,Ny+1,Nz),order='F')
    p = a[wend:].reshape((Nx+1,Ny+1,Nz+1),order='F')
    return u, v, w, p, iters, time

time = timeit.default_timer()
#Parameters   
Re = 400
nt = 10000
ntrest = 100
ntdsk = 50
restart = False
Nx = 32
Ny = 32
Nz = 32
N = Nx*Ny*Nz
Lx = 1.0
Ly = 1.0
Lz = 1.0
hx = Lx/(Nx-1)
Ly = hx*(Ny-1) # Ensuring uniform mesh
hy = Ly/(Ny-1)
hz = Lz/(Nz-1)
h = hx # Assuming uniform mesh
#Generate grid for cell edges
xe = np.linspace(0,Lx,Nx)
ye = np.linspace(0,Ly,Ny)
ze = np.linspace(0,Lz,Nz)
x, y, z = np.meshgrid(xe, ye, ze, indexing='ij')
#Generate grid for cell centers
xc = np.linspace(-hx/2,Lx+hx/2,Nx+1)
yc = np.linspace(-hy/2,Ly+hy/2,Ny+1)
zc = np.linspace(-hz/2,Lz+hz/2,Nz+1)
#Initialise variables
u = np.zeros((Nx,Ny+1,Nz+1))
v =  np.zeros((Nx+1,Ny,Nz+1))
w = np.zeros((Nx+1,Ny+1,Nz))
p = np.zeros((Nx+1,Ny+1,Nz+1))
#boundary conditions
u, v, w, p = updateBC(u,v,w,p)
#Start time and iteration
t = 0
iteration = 0
#restart
if restart:
    re_file = open("C:/Users/debar/Desktop/python_tecplot/ldc33res_10.dat",'r')
    u, v, w, p, iteration, t = readdata(re_file)
    re_file.close()
#Information for multigrid
prolong_memory = [[],[]]
restrict_memory = [[],[]]
numLevels = 5
ijkLevels = np.array([[33, 17, 9, 5, 3],\
                      [33, 17, 9, 5, 3],\
                      [33, 17, 9, 5, 3]])
A = generate_A(ijkLevels)

while iteration < nt:
    dt = timestep(u,v,w)
    t += dt
    iteration += 1
    u_old = u.copy()
    v_old = v.copy()
    w_old = w.copy()
    Eu, Ev, Ew = explicit(u,v,w)
    qu = dt*Eu
    qv = dt*Ev
    qw = dt*Ew
    u, v, w, p, qu, qv, qw = rk(0, 1/3, 1/3, dt, u, v, w, p, qu, qv, qw, Eu, Ev, Ew)
    u, v, w, p = updateBC(u,v,w,p)
    Eu, Ev, Ew = explicit(u,v,w)
    u, v, w, p, qu, qv, qw = rk(-5/9, 15/16, 5/12, dt, u, v, w, p, qu, qv, qw, Eu, Ev, Ew)
    u, v, w, p = updateBC(u,v,w,p)
    Eu, Ev, Ew = explicit(u,v,w)
    u, v, w, p, qu, qv, qw = rk(-153/128, 8/15, 1/4, dt, u, v, w, p, qu, qv, qw, Eu, Ev, Ew)
    u, v, w, p = updateBC(u,v,w,p)
    error = residual(Eu, Ev, Ew, u, v, w, p, u_old, v_old, w_old, dt)
#    res_file = open("C:/Users/debar/Desktop/python_tecplot/residual.txt",'a')
#    res_file.write('Iteration = ' + str(i+1) + ', time = ' + str(t) + \
#          ', CPU time = ' + str(timeit.default_timer()-time) + ', |Residual| = '\
#          + str(error) + '\n')
#    res_file.close()
    print('Iteration = ' + str(iteration) + ', time = ' + str(t) + \
          ', CPU time = ' + str(timeit.default_timer()-time) + ', |Residual| = '\
          + str(error))
    if iteration % ntdsk == 0:
        un, vn, wn, pn = convertToNodalData(u,v,w,p)
        savedata(x,y,z,un,vn,wn,pn,Nx,Ny,Nz,iteration)
    if iteration % ntrest == 0:
        saverestartdata(u,v,w,p,iteration,t)
        

        