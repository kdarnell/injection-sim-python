#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import flashalgorithm as fc
import numpy as np
import pickle
import pdb

comp_list = ('water', 'methane', 'ethane', 'propane')
phase_list = ('vapor', 'lhc')
P = 70  # bar
T = 273.15 + 6 # Kelvin

flash_full = fc.FlashController(components=comp_list,
                                phases=phase_list)

def emptycomp_hash(z):
    hashed = sum([2**ii for ii, x in enumerate(z) if x == 0.0])
    return hashed

z_use = pickle.load(open('70bar6C_complist.pkl', 'rb'))
flash_dict = {0: flash_full}
all_output = list()
out_file = 'c1toc3_flashtable_70bar6C_nowater_quick.pkl'
K_use = []
for ii, z in enumerate(z_use):
    comp_hash = emptycomp_hash(z)

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
    try:
        output = flash_use.main_handler(
                    compobjs=flash_use.compobjs,
                    z=new_z,
                    T=T,
                    P=P,
                    initialize=True)
        if output[-1] > 1e-6:
            output = flash_use.main_handler(
                compobjs=flash_use.compobjs,
                z=new_z,
                T=T,
                P=P,
                initialize=True,
                incipient_calc=True)
    except:
        try:
            output = flash_use.main_handler(
                        compobjs=flash_use.compobjs,
                        z=new_z,
                        T=T,
                        P=P,
                        initialize=True,
                        incipient_calc=False)
            if output[-1] > 1e-6:
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
    if np.mod(ii, 10) == 0:
	#pdb.set_trace()
        print('{0:3.3f} % complete!'.format(float(ii) * 100 / len(z_use)))
        with open(out_file, 'wb') as f:
            pickle.dump(all_output, f)

with open(out_file, 'wb') as f:
     pickle.dump(all_output, f)

