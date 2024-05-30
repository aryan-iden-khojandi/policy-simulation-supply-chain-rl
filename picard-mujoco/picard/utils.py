from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def iterate(f, x, n):
    """
    Computes x, f(x), f(f(x)), ..., f^n(x)
    Returns a tuple of the final value and the intermediate values.
    """
    def iterate_f(x, _):
        return f(x), x
    return jax.lax.scan(iterate_f, x, None, n)


def iterate_until(f: Callable, x: Any, cond: Callable) -> Tuple[int, Any]:
    """
    Computes x, f(x), f(f(x)), ..., f^n(x)
    until some condition cond(n, f^{n-1}(x), f^{n}(x)) is met,
    or max_iters iterations are reached.
    Returns a value of the same type as x
    """
    def iterate_f(xs):
        it, x, fx = xs
        return (it + 1, fx, f(fx))

    def cond_f(xs):
        return jnp.logical_not(cond(*xs))

    result = jax.lax.while_loop(cond_f, iterate_f, (0, x, f(x)))
    return result[2], result[0]


def accumulate(f, xs, init):
    """
    Given xs: List[a], init: b, and f: (b, a) -> b,

    """
    def accumulate_f(b, a):
        out = f(b, a)
        return out, out
    return jax.lax.scan(accumulate_f, init, xs)


def max_error_between_trees(x, y, reduce=max):
    traj_errors = jax.tree.map(
        lambda a, b: jnp.abs(a.astype(float) - b.astype(float)).max(),
        x, y
    )
    if reduce is not None:
        return jax.tree.reduce(reduce, traj_errors, 0.)
    return traj_errors
