"""
This script is used to compose a collection of files describing the electron configurations and
binding energies of elements in different charge states into a single file for simplified import
into the ebisim package.

The original data was computed by Robert Mertzig using the Flexible Atomic Code
(https://github.com/flexible-atomic-code/fac)
"""

import json
import time


# From Roberts readme:

# X.txt contains binding energies for the different (sub)shells for element(Z=X)
# Xconf.txt contains (sub)shell occupation
# 1s 2s 2p- 2p+ 3s 3p- 3p+ 3d- 3d+ 4s 4p- 4p+ 4d- 4d+ 5s 5p- 5p+ 4f- 4f+ 5d- 5d+ 6s 6p- 6p+ 5f- 5f+ 6d- 6d+ 7s 
# 0  1  2   3   4  5   6   7   8   9  10  11  12  13  14 15  16  17  18  19  20  21 22  23  24  25  26  27  28

# the readme is missing (for the case z=103 Lr)
# 7p-
# 29
# This somewhat arbitrary order should be rearranged

REDICT = {
    0 : 0,
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 4,
    5 : 5,
    6 : 6,
    7 : 7,
    8 : 8,
    9 : 9,
    10 : 10,
    11 : 11,
    12 : 12,
    13 : 13,
    14 : 16,
    15 : 17,
    16 : 18,
    17 : 14,
    18 : 15,
    19 : 19,
    20 : 20,
    21 : 23,
    22 : 24,
    23 : 25,
    24 : 21,
    25 : 22,
    26 : 26,
    27 : 27,
    28 : 28,
    29 : 29
}

def reorder(l):
    """
    This method uses the above dictionary to reorder a list in such a way, that it corresponds to
    the shells being sorted by n, then by the angular momentum and then by the coupling - < +

    E.g.
    >>> reorder(['1s', '2s', '2p-', '2p+', '3s', '3p-', '3p+', '3d-', '3d+', '4s', '4p-', '4p+', '4d-', '4d+', '5s', '5p-', '5p+', '4f-', '4f+', '5d-', '5d+', '6s', '6p-', '6p+', '5f-', '5f+', '6d-', '6d+', '7s'])
    ['1s', '2s', '2p-', '2p+', '3s', '3p-', '3p+', '3d-', '3d+', '4s', '4p-', '4p+', '4d-', '4d+', '4f-', '4f+', '5s', '5p-', '5p+', '5d-', '5d+', '5f-', '5f+', '6s', '6p-', '6p+', '6d-', '6d+', '7s']
    """
    maxind = max(map(REDICT.get, range(len(l))))
    out = [0 for _ in range(maxind+1)]
    for i, val in enumerate(l):
        out[REDICT[i]] = val
    return out

def load_conf(z):
    # Import Electron Configurations for each charge state
    # list of lists where each sublist hold the configuration for on charge state
    # cfg[n] describes charge state n+
    cfg = []
    n_cfg_max = 0
    with open(f"./resources/BindingEnergies/{z}conf.txt") as fobj:
        for line in fobj:
            line = line.split()
            line = reorder([int(elem.strip()) for elem in line])
            n_cfg_max = max(n_cfg_max, len(line))
            cfg.append(line)
    return cfg

def load_energies(z):
    # Load required data from resource files, can set further fields
    # Import binding energies for each electron in all charge states
    # list of lists where each sublist hold the energies for one charge state
    # e_bind[n] describes charge state n+
    e_bind = []
    with open(f"./resources/BindingEnergies/{z}.txt") as fobj:
        for line in fobj:
            line = line.split()
            line = reorder([float(elem.strip()) for elem in line])
            e_bind.append(line)
    return e_bind



def main():
    out = {}
    for z in range(1, 106):
        data = {}
        data["ebind"] = load_energies(z)
        data["conf"] = load_conf(z)
        out[str(z)] = data # json keys are always strings
    
    with open("../ebisim/resources/BindingEnergies.json", "w") as f:
        json.dump(out, f)
    
    start = time.time()
    with open("../ebisim/resources/BindingEnergies.json", "r") as f:
        val = json.load(f)
    print(f"Loading took {time.time() - start} s.")

    print("Valid" if out == val else "Invalid")

if __name__ == "__main__":
    main()


        # # This block could be useful in the future for parallelising cross section computations
        # e_bind_mat = np.zeros((n_cfg_max, self.element.z+1))
        # cfg_mat = np.zeros((n_cfg_max, self.element.z+1))
        # for cs, data in enumerate(e_bind):
        #     e_bind_mat[:len(data), cs] = data
        # for cs, data in enumerate(cfg):
        #     cfg_mat[:len(data), cs] = data

        # e_bind_min = e_bind[0][-1]
        # e_bind_max = e_bind[-1][0]