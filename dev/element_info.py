"""
Reads atomic number Z, symbol, and name information from ChemicalElements.csv and
writes a json resource
"""
import json
import time

def main():
    print("element_info.py running...")

    z = [] # Atomic Number
    a = [] # Mass Number
    es = [] # Element Symbol
    name = [] # Element Name
    ip = []
    with open("./resources/ChemicalElements.csv") as f:
        f.readline() # skip header line
        for line in f:
            data = line.split(",")
            z.append(int(data[0].strip()))
            es.append(data[1].strip())
            name.append(data[2].strip())
            a.append(int(data[3].strip()))
            ip.append(float(data[4].strip()))

    out = dict(z=z[:105], a=a[:105], es=es[:105], name=name[:105], ip=ip[:105])

    with open("../ebisim/resources/ElementInfo.json", "w") as f:
        json.dump(out, f)

    start = time.time()
    with open("../ebisim/resources/ElementInfo.json", "r") as f:
        val = json.load(f)
    print(f"Loading took {time.time() - start} s.")

    print("json.load() valid" if out == val else "json.load() invalid")
    print("element_info.py done.")

if __name__ == "__main__":
    main()
