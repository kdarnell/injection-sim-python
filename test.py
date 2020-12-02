#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import flashalgorithm as fc
import numpy as np
import pickle
import pdb

comp_list = ('water', 'methane', 'ethane', 'propane')
phase_list = ('aqueous', 'vapor', 'lhc', 's1', 's2')
P = 70  # bar
T = 273.15 + 6 # Kelvin

flash_full = fc.FlashController(components=comp_list,
                                phases=phase_list)

water_fracs = [0.8, 0.9]
hc_fracs = np.linspace(0, 1, 20)
c1_frac, c2_frac, c3_frac = np.meshgrid(hc_fracs, hc_fracs, hc_fracs)


z_all = list()
for water in water_fracs:
    for hcs in zip(c1_frac.flatten(), c2_frac.flatten(), c3_frac.flatten()):
        if sum(hcs) > 0.0:
            mod_hcs = [x / sum(hcs) * (1.0 - water) for x in hcs]
            z = np.asarray([water] + mod_hcs)
            z_all.append(z / np.sum(z))
z_use = np.unique(np.asarray(z_all), axis=0)


def emptycomp_hash(z):
    hashed = sum([2**ii for ii, x in enumerate(z) if x == 0.0])
    return hashed

flash_dict = {0: flash_full}
all_output = list()
out_file = 'c1toc3_flashtable_70bar6C_pt3.pkl'
for ii, z in enumerate(z_use):
    comp_hash = emptycomp_hash(z)
    K_use = []
    if ii > 0:
        z_prev = z_use[:ii, :]
        water_same = z_prev[:, 0] == z[0]
        if water_same.any():
            comp_same = [ii for ii, z_iter in enumerate(z_prev) if comp_hash == emptycomp_hash(z_iter)]
            if len(comp_same) > 0:
                z_tmp, out_tmp = (
                    z_prev[water_same, :],
                    [x for ii, (x, check) in enumerate(zip(all_output, water_same))
                     if (check) and (ii in comp_same)]
                )
                min_ind = np.sum((z[np.newaxis, :] - z_tmp)**2).argmin()
            try:
                K_use = out_tmp[min_ind][4][2]
            except:
                K_use = []

    new_comps, new_z = zip(*[
        (comp, z_) for comp, z_ in zip(comp_list, z)
        if z_ != 0.0
    ])
    if comp_hash not in flash_dict.keys():
        flash_dict.update({comp_hash:
                               fc.FlashController(
                                    components=new_comps,
                                    phases=phase_list)})
    flash_use = flash_dict[comp_hash]
    new_z = np.asarray(new_z)
    pdb.set_trace()

    try:
        if K_use != []:
            output = flash_use.main_handler(
                        compobjs=flash_use.compobjs,
                        z=new_z,
                        T=T,
                        P=P,
                        initialize=True,
                        K_init=K_use)
        else:
            output = flash_use.main_handler(
                        compobjs=flash_use.compobjs,
                        z=new_z,
                        T=T,
                        P=P,
                        initialize=True)
    except:
        try:
            output = flash_use.main_handler(
                        compobjs=flash_use.compobjs,
                        z=new_z,
                        T=T,
                        P=P,
                        initialize=True,
                        incipient_calc=True)
        except:
            output = []


    all_output.append([ii, z, new_comps, new_z, output])
    if np.mod(ii, 100) == 0:
        pdb.set_trace()
        print('{0:3.3f} % complete!'.format(float(ii) * 100 / len(z_use)))
        with open(out_file, 'wb') as f:
            pickle.dump(all_output, f)

with open(out_file, 'wb') as f:
     pickle.dump(all_output, f)

