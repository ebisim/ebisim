import ebisim as eb
import numpy as np

for z in range(1,106):
    chel = eb.ChemicalElement(z)
    eixs = eb.EIXS(z)
    el = eb.Element(z)
    for e in [500, 5000, 50000]:
        xs = eixs.xs_vector(e)
        xs2 = eb._eix_xs_vector_2(el, e)
        if not np.allclose(xs, xs2, atol=0, rtol=1.e-10):
            print(el.name, e)
    print(f"Checked {el.name}")