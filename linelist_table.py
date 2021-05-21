import ChiantiPy.core as ch
import numpy as np
import matplotlib.pyplot as plt
import pandas
import astropy.units as u
from astropy.io import ascii

# The goal of this test file is to match the results of Chiantipy to that of ch_ss in IDL to make sure we know how
# everything works.

if __name__ == '__main__':
    dem_file = '/home/jake/chianti/dem/quiet_sun.dem'
    dem = pandas.read_csv(dem_file, sep=' ', skipinitialspace=True, skipfooter=9, names=['logT', 'EM'])
    wvl_range = [580, 630]

    dem_logT = dem['logT'].to_numpy()
    temperature = 10 ** dem_logT
    dlnt = np.log(10 ** (dem_logT[1] - dem_logT[0]))
    dt = temperature * dlnt


    pressure = 1e15
    dens = pressure / temperature
    em = 10 ** dem['EM'].to_numpy()
    ion_list = ['o_3', 'o_4', 'o_5', 'he_1', 'mg_10']
    abund_file = 'sun_coronal_2012_schmelz'
    minabund = 7.4e-5

    test_bunch = ch.bunch(temperature, dens, wvl_range, em=em, abundance=abund_file,
                          allLines=0,
                          verbose=True, ionList=ion_list,
                          # minAbund=minabund,  # Note, minAbund will override ionList
                          )

    # test_bunch.intensityList(wvlRange=wvl_range,top=5,)
    # print(test_bunch.Intensity)
    wvl = test_bunch.Intensity['wvl']
    mask = (wvl < wvl_range[1]) & (wvl > wvl_range[0])
    wvl = wvl[mask]
    unsummed_int = test_bunch.Intensity['intensity'][:,mask]
    ions = test_bunch.Intensity['ionS'][mask]

    ints = np.sum(unsummed_int*dt[...,None],axis=0)
    sort = wvl.argsort()
    # sort = sort[::-1]
    top_lines = 7
    intensity_mask = ints[sort] > np.sort(ints)[::-1][top_lines]

    sorted_ions = ions[sort][intensity_mask]
    print(sorted_ions)
    sorted_wvls = wvl[sort][intensity_mask]
    print(sorted_wvls)
    sorted_ints = ints[sort][intensity_mask] / ints.max()
    print(sorted_ints)

    ascii.write([sorted_ions,sorted_wvls.round(decimals=2),sorted_ints.round(decimals=2)], 'linelist_table.tex',format='latex', overwrite=True)
