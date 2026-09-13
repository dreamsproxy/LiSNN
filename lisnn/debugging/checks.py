"""Small reporting helpers shared by independently callable smoke tests."""


def report(checks, verbose=True):
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(checks) and all(checks.values())
    if verbose:
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'} {name}")
    return {"passed": passed, "checks": checks}


def rejects(exception, function, *args, **kwargs):
    """Return False if invalid input was accepted; unexpected errors propagate."""
    try:
        function(*args, **kwargs)
    except exception:
        return True
    return False
