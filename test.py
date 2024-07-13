import numpy as np
def calc_pot_aparente(flag, x, pf):
    if flag == "kW":
        return x / pf
    elif flag == "kVA":
        return x
    else:  # kVar = kVA

        return x
    


carga_c1 = calc_pot_aparente("kW", 2000, 1)
carga_c2 = calc_pot_aparente("kW", 800, 0.92)
carga_c3 = calc_pot_aparente("kvar", 800, None)

carga_d = calc_pot_aparente("kW", 1000, 0.92)

carga_total = carga_c1 + carga_c2 + carga_c3 + carga_d
print(f"Carga total: {carga_total:.2f} kVA")