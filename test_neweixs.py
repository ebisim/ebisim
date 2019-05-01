import ebisim as eb
import numpy as np

for z in range(1,106):
    chel = eb.ChemicalElement(z)
    eixs = eb.EIXS(z)
    el = eb.Element(z)
    for e in [500, 5000, 50000]:
        xs = eixs.xs_vector(e)
        xs2 = eb._eixs_vector_2(el, e)
        if not np.allclose(xs, xs2, atol=0, rtol=1.e-10):
            print("XXXXX EI", el.name, e)
    print(f"EI Checked {el.name}")

for z in range(1,106):
    chel = eb.ChemicalElement(z)
    rrxs = eb.RRXS(z)
    el = eb.Element(z)
    for e in [500, 5000, 50000]:
        xs = rrxs.xs_vector(e)
        xs2 = eb._rrxs_vector_2(el, e)
        if not np.allclose(xs, xs2, atol=0, rtol=1.e-10):
            print("XXXXX RR", el.name, e)
    print(f"RR Checked {el.name}")

for z in range(1,106):
    chel = eb.ChemicalElement(z)
    drxs = eb.DRXS(z)
    el = eb.Element(z)
    for e in [500, 5000, 50000, el.dr_e_res[0] if el.dr_e_res.size else 0]:
        for fwhm in [10,25]:
            xs = drxs.xs_vector(e, fwhm)
            xs2 = eb._drxs_vector_2(el, e, fwhm)
            if not np.allclose(xs, xs2, atol=0, rtol=1.e-10):
                print("XXXXX DR", el.name, e, fwhm)
    print(f"DR Checked {el.name}")