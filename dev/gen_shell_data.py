"""
This script is used to compose a collection of files describing the electron configurations and
binding energies of elements in different charge states into a single file for simplified import
into the ebisim package.

The original data was computed by Robert Mertzig using the Flexible Atomic Code
(https://github.com/flexible-atomic-code/fac)
"""
import os
import json
import time
from shutil import move
from string import Template


# From Roberts readme:

# X.txt contains binding energies for the different (sub)shells for element(Z=X)
# Xconf.txt contains (sub)shell occupation
# 1s 2s 2p- 2p+ 3s 3p- 3p+ 3d- 3d+ 4s 4p- 4p+ 4d- 4d+ 5s 5p- 5p+ 4f- 4f+ 5d- 5d+ 6s 6p- 6p+ 5f- 5f+ 6d- 6d+ 7s
# 0  1  2   3   4  5   6   7   8   9  10  11  12  13  14 15  16  17  18  19  20  21 22  23  24  25  26  27  28

# the readme is missing (for the case z=103 Lr)
# 7p-
# 29
# This somewhat arbitrary order should be rearranged

SHELLS_IN = ('1s', '2s', '2p-', '2p+', '3s', '3p-', '3p+', '3d-', '3d+', '4s', '4p-', '4p+', '4d-', '4d+', '5s', '5p-', '5p+', '4f-', '4f+', '5d-', '5d+', '6s', '6p-', '6p+', '5f-', '5f+', '6d-', '6d+', '7s', '7p-')

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
    >>> reorder(['1s', '2s', '2p-', '2p+', '3s', '3p-', '3p+', '3d-', '3d+', '4s', '4p-', '4p+', '4d-', '4d+', '5s', '5p-', '5p+', '4f-', '4f+', '5d-', '5d+', '6s', '6p-', '6p+', '5f-', '5f+', '6d-', '6d+', '7s', '7p-'])
    ['1s', '2s', '2p-', '2p+', '3s', '3p-', '3p+', '3d-', '3d+', '4s', '4p-', '4p+', '4d-', '4d+', '4f-', '4f+', '5s', '5p-', '5p+', '5d-', '5d+', '5f-', '5f+', '6s', '6p-', '6p+', '6d-', '6d+', '7s', '7p-']
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
    with open(f"./resources/BindingEnergies/{z}conf.txt") as fobj:
        for line in fobj:
            line = line.split()
            line = reorder([int(elem.strip()) for elem in line])
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
    print(30*"~")
    print(f"{__name__} running...")

    CWD = os.getcwd()
    TWD = os.path.dirname(os.path.realpath(__file__))
    print(f"Switching into {TWD}")
    os.chdir(TWD)

    blocks = []
    print("Loading data from electron configuration files.")
    out = {}
    for z in range(1, 106):
        data = {}
        data["ebind"] = load_energies(z)
        data["cfg"] = load_conf(z)
        out[str(z)] = data # json keys are always strings
        blocks.append(
            BLOCKTEMPLATE.substitute(z=z, ebind=repr(data["ebind"]), cfg=repr(data["cfg"]))
            )

    ORDER = tuple(reorder(SHELLS_IN))
    N = tuple((map(int, [s[0] for s in ORDER])))

    out = FILETEMPLATE.substitute(order=repr(ORDER), data="".join(blocks), n=N)
    # Write file
    print("Writing output file.")
    with open("temp_shell_data.py", "w") as f:
        f.write(out)

    # Check if json load correctly restores the output
    # print("Peforming test import.")
    # start = time.time()
    # with open("../ebisim/resources/BindingEnergies.json", "r") as f:
    #     val = json.load(f)[1]
    # print(f"Test import took {time.time() - start} s.")


    # print("Test import valid!" if valid else "Test import invalid!")

    print("Moving output file to target location ebisim/resources.")
    move("temp_shell_data.py", "../ebisim/resources/_shell_data.py")

    print(f"Returning into {CWD}")
    os.chdir(CWD)

    print("binding_energies.py done.")
    print(30*"~")

FILETEMPLATE = Template('''"""
This module contains data concerning the electron configuration and binding energies used all
throughout ebisim
"""

# This file is generated automatically, do not edit it manually!

ORDER = $order

N = $n

DATA = {
$data
}
''')

BLOCKTEMPLATE = Template('''    $z:{
        "ebind":$ebind,
        "cfg":$cfg,
    },
''')



if __name__ == "__main__":
    main()
