def zip_equal(it1, it2):
    if len(it1) != len(it2):
        raise ValueError("Lengths of iterables are different")
    return zip(it1, it2)