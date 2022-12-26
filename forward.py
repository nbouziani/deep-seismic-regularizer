import ufl
from firedrake import *

from collections.abc import Sequence
import numbers


class ForwardModel(object):

    def __init__(self, V, solver_parameters=None):
        # Test both
        self.V = V
        self.solver_params = solver_parameters or {'ksp_type': 'gmres'}

    def __call__(self, c, annotate=True, tape=None):
        raise NotImplementedError


class WaveForward(ForwardModel):

    def __init__(self, f, V, T, dt, solver_parameters=None):
        super().__init__(V, solver_parameters)

        # Add check
        if isinstance(f, ufl.Expr):
            raise ValueError("Expecting a UFL expression and not %s" % str(type(f)))

        self.f = f

        if not isinstance(T, numbers.Real):
            raise TypeError("T must be a real number and not %s" % str(type(T)))
        if not isinstance(dt, numbers.Real):
            raise TypeError("dt must be a real number and not %s" % str(type(dt)))

        self.T = T
        self.dt = dt

    def __call__(self, c, annotate=True, tape=None):
        V = self.V
        solver_params = self.solver_params
        T = self.T
        dt = self.dt
        f = self.f

        # Solve Wave equation
        u = TrialFunction(V)
        v = TestFunction(V)

        t = 0
        p = Function(V, name="p")
        φ = Function(V, name="φ")
        while t <= T:
            φ -= dt / 2 * p

            rhs = v * p * dx + dt * inner(grad(v), c**2 * grad(φ)) * dx
            if t <= 10 * dt:
                rhs += inner(f, v) * dx
            solve(u * v * dx == rhs, p, solver_parameters=solver_params)
            φ -= dt / 2 * p
            t += dt
        return φ


class WaveForwardTraining(ForwardModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, c, annotate=True, tape=None):
        pass


class WaveguideForward(ForwardModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, c, annotate=True, tape=None):
        pass

class MarmousiForward(ForwardModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, c, annotate=True, tape=None):
        pass
