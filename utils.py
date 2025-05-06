def compose(f, g):
    return lambda x: f(g(x))


def identity(x):
    return x
