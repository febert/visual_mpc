def zip_equal(*args):
    lengths = [len(el) for el in args]
    if not all(el == lengths[0] for el in lengths):
        raise ValueError("Lengths of iterables are different")
    return list(zip(*args))