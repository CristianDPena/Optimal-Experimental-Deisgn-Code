import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from InitialFPsolver.utils import remove_duplicates
from InitialFPsolver.utils import solve_lu
import time

day = 24*60*60 #seconds

# -------------------------------------------------------
# 1) Problem and Numerical Parameters
# -------------------------------------------------------

# Fokker-Planck coefficients
alpha = 1.0  # Ornstein-Uhlenbeck "spring" constant
D = 1.0e-10      # Diffusion coefficient (1/s)
T_end = 100*day



def diffCoeff(x):
    return D*x**3.

def sourceLoss(x):
    return 0

# -------------------------------------------------------
# 2) Initial Condition
# -------------------------------------------------------

def load_single_orbit(filename, N):
    """Load data, remove duplicates, sort, then
    interpolate onto a uniform grid of length N."""
    # 1. Load raw data
    xin, fin = np.loadtxt(filename,
                          usecols=[1, 2],
                          unpack=True,
                          skiprows=1)

    # 2. Remove duplicates
    xin, fin = remove_duplicates(xin, fin)

    # 3. Sort so xin is strictly increasing
    sort_idx = np.argsort(xin)
    xin = xin[sort_idx]
    fin = fin[sort_idx]

    # 4. Create uniform grid and interpolate f
    x = np.linspace(xin.min(), xin.max(), N)
    intp = CubicSpline(xin, fin)
    f = intp(x)
    return x, f



#f = gaussian(x, mu=0.0, sigma=0.8)

Nx = 200
x, f = load_single_orbit("singleOrbit_t50_10MeV.txt", Nx)
f0 = np.array(f)

f*=0

f[-1] = np.exp(32.)
f[0] = np.exp(21.)

fprime_left = -1.*f[0]
fprime_right = 0.

# Domain: x in [x_min, x_max]
dx = x[1] - x[0]


## Impose simple timestep constraint
def compute_max_timestep(x):
    dt_inv = diffCoeff(x)/(dx**2.)
    return np.min(0.25/dt_inv)


# Time parameters
dt = compute_max_timestep(x)
nt = int(T_end / dt)

# -------------------------------------------------------
# 3) Discretizing the Fokker-Planck Operator
# -------------------------------------------------------
# PDE: partial_t f = alpha d/dx(x f) + D d^2 f/dx^2
#
# We'll construct a linear operator L such that:
# L f_j = alpha * d/dx(x_j * f_j) + D * d^2 f_j / dx^2
#
# Then the Crank–Nicolson step is:
# (I - 0.5*dt*L) f^{n+1} = (I + 0.5*dt*L) f^{n}

# For i in [1..Nx-2], define finite differences:
# 1) Diffusion term (D d^2/dx^2):
#    (f_{j-1} - 2 f_j + f_{j+1}) / dx^2
# 2) Drift term (alpha d/dx[x f]):
#    alpha * [ (x_{j+1} * f_{j+1} - x_{j-1} * f_{j-1}) / (2 dx) ],
#    using central differences for the derivative.

# We'll build the Nx x Nx matrices for the interior,
# but typically we apply boundary conditions at j=0, j=Nx-1.


start_time = time.time()

# Initialize empty Nx x Nx matrices (we'll keep them small for clarity).
L_matrix = np.zeros((Nx, Nx))

for j in range(1, Nx-1):
    # Indices: j-1, j, j+1
    xm = x[j-1]
    x0 = x[j]
    xp = x[j+1]

    xp0 = xp/x0
    xm0 = xm/x0

    # -- Diffusion part: D * second difference
    L_matrix[j, j-1] += xm0*xm0*diffCoeff(xm)/ dx**2
    L_matrix[j, j]   -= (xm0*xm0*diffCoeff(xm) + xp0*xp0*diffCoeff(xp))/ dx**2
    L_matrix[j, j+1] += xp0*xp0*diffCoeff(xp)/ dx**2

    # -- Drift part: alpha * derivative of (x_j * f_j)
    #    Approximated by central difference:
    #    d/dx( x f ) ~ [ (x_{j+1} f_{j+1}) - (x_{j-1} f_{j-1}) ] / (2 dx)

    # Contribution from x_{j+1} f_{j+1} in the derivative at j
    L_matrix[j, j+1] += sourceLoss(xp) / (2*dx)

    # Contribution from - x_{j-1} f_{j-1} in the derivative at j
    L_matrix[j, j-1] += -sourceLoss(xm) / (2*dx)

# We will apply **Dirichlet boundary conditions**: f( x_min ) = f( x_max ) = 0
# (If you prefer no-flux (Neumann) or periodic, you must modify accordingly.)
# For Dirichlet = 0, we can enforce by zeroing out the boundary rows and
# setting the diagonal to 1, effectively holding f(0)=f(Nx-1)=0.


### IMPORTANT: NEED CONSTANT FLUX BOUNDARIES

# Keeps boundaries fixed
L_matrix[0, :] = 0.0
L_matrix[-1, :] = 0.0

L_matrix[1, :] = 0.0
#L_matrix[-2, :] = 0.0

#Need to enforce a fixed gradient
#L_matrix[1, 1] =  diffCoeff(x[1])/(dx*dx)
#L_matrix[1, 0] = -diffCoeff(x[1])/(dx*dx)

#L_matrix[-2, -2] = -diffCoeff(x[-2])/(dx*dx)
#L_matrix[-2, -1] = +diffCoeff(x[-2])/(dx*dx)

#L_matrix[-1, -1] = +0.5*diffCoeff(x[-1])/(dx*dx)
#L_matrix[-1, -3] = -0.5*diffCoeff(x[-3])*x[-1]*x[-1]/(x[-3]*x[-3]*dx*dx)

# -------------------------------------------------------
# 4) Crank–Nicolson Matrices
# -------------------------------------------------------
#   A = I - 0.5 dt L
#   B = I + 0.5 dt L
I_matrix = np.eye(Nx)
A = I_matrix - dt * L_matrix

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print("Fill matrix:", elapsed_time, "seconds")

from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu
#lu, piv = lu_factor(A)

start_time = time.time()

Alu = splu(A)

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print("Compute LU:", elapsed_time, "seconds")

# Pre-factorizing A could be useful if Nx is large; here we just use solve.
# E.g. with SciPy:
#   then solve each step with lu_solve((lu, piv), B @ f)


start_time = time.time()

# -------------------------------------------------------
# 5) Time-stepping
# -------------------------------------------------------
for n in range(nt):
    # Solve for f^{n+1}:  A f^{n+1} = rhs
#   dd f_new = lu_solve((lu, piv), f)
#    f_new = solve_lu(lu, piv, f)
    f_new = Alu.solve(f)

    ## need to include boundary loss
    #f_new[1]  -= dt/dx*x[1]/x[0]*diffCoeff(x[0])*fprime_left
    #f_new[-2] += dt/dx*x[-2]/x[-2]*diffCoeff(x[-1])*fprime_right

    #f_new = np.linalg.solve(A, f)
    f = f_new

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print("Integration:", elapsed_time, "seconds")

# -------------------------------------------------------
# 6) Plot the Result
# -------------------------------------------------------
plt.figure(figsize=(6,4))
plt.semilogx(x, np.log(np.abs(f0)), label='Initial distribution')
plt.semilogx(x, np.log(np.abs(f)), label='Final distribution')
plt.xlabel('L')
plt.ylabel('log f(x, t)')
plt.grid(True)
plt.legend()
plt.show()

