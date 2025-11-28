def MVD(lt, cas, q, L, I):
    """Maximum moment, shear force, and deflection of a beam.
    Parameters:
        lt (str): load type ('uniform' or 'double ponctuel')
        cas (str): case ('y' or 'z')
        q (float): load [MN/m or force unit consistent with formulas]
        L (float): beam length [m]
        I (float): moment of inertia [m^4]
    Returns:
        Mmax (float): maximum moment
        Vmax (float): maximum shear
        deltamax (float): maximum deflection [m]
    """
    # Steel Young's modulus
    E = 2 * 10**5  # MPa = N/mm² (be careful with unit consistency!)
    if lt == "uniform":
        if cas == "y":
            Mmax = q * L**2 / 8
            Vmax = q * L / 2
            deltamax = (5/384) * (q * L**4) / (E * I)
        elif cas == "z":
            Mmax = q * L**2 / 32
            Vmax = 0.625 * q * L / 2
            deltamax = 0.0
    elif lt == "double ponctuel":
        a = L / 3
        if cas == "y":
            Mmax = q * a
            Vmax = q
            deltamax = q * a * (3*L**2 - 4*a**2) / (24 * E * I)
        elif cas == "z":
            Mmax = 0.0926 * q * L
            Vmax = 0.852 * q
            deltamax = 0.0
    else:
        raise ValueError("Unknown load type")
    return Mmax, Vmax, deltamax
