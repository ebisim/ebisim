"""
Reads atomic number Z, symbol, and name information from ChemicalElements.csv and
writes a json resource
"""
import os
import time
from shutil import move


def main():
    CWD = os.getcwd()
    TWD = os.path.dirname(os.path.realpath(__file__))
    print(30*"~")
    print(f"{__name__} running...")
    print(f"Switching into {TWD}")
    os.chdir(TWD)

    print("Loading data from ChemicalElements.csv")
    z = []  # Atomic Number
    es = []  # Element Symbol
    name = []  # Element Name
    a = []  # Mass Number
    ip = []
    with open("./resources/ChemicalElements.csv") as f:
        f.readline()  # skip header line
        for line in f:
            data = line.split(",")
            z.append(int(data[0].strip()))
            es.append(data[1].strip())
            name.append(data[2].strip())
            a.append(int(data[3].strip()))
            ip.append(float(data[4].strip()))

    out = dict(Z=z[:105], A=a[:105], ES=es[:105], NAME=name[:105], IP=ip[:105])

    DOC = '"""This module contains several tuples used by the elements module in ebisim"""\n\n'
    AUTO = "# This file is generated automatically, do not edit it manually!\n\n"
    lines = list((DOC, AUTO))
    for k, v in out.items():
        lines.append(f"{k} = (\n")
        for e in v:
            lines.append(f"    {repr(e)},\n")
        lines.append(")\n")
        lines.append("\n")
    lines.pop(-1)  # discard trailing newline

    print("Writing output file.")
    with open("temp_element_data.py", "w") as f:
        f.writelines(lines)

    print("Peforming test import.")
    start = time.time()
    from temp_element_data import Z, ES, NAME, A, IP
    print(f"Test import took {time.time() - start} s.")

    valid = all([
        tuple(z) == Z,
        tuple(es) == ES,
        tuple(name) == NAME,
        tuple(a) == A,
        tuple(ip) == IP
    ])
    print("Test import valid!" if valid else "Test import invalid!")

    print("Moving output file to target location ebisim/resources.")
    move("temp_element_data.py", "../ebisim/resources/_element_data.py")

    print(f"Returning into {CWD}")
    os.chdir(CWD)

    print(f"{__name__} done.")
    print(30*"~")


if __name__ == "__main__":
    main()
