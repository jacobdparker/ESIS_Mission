import ChiantiPy.core as ch
import numpy as np
import matplotlib.pyplot as plt
import pandas
import astropy.units as u
from astropy.io import ascii

# The goal of this test file is to match the results of Chiantipy to that of ch_ss in IDL to make sure we know how
# everything works.

from collections import OrderedDict

def write_roman(num):

    roman = OrderedDict()
    roman[40] = "xl"
    roman[10] = "x"
    roman[9] = "ix"
    roman[5] = "v"
    roman[4] = "iv"
    roman[1] = "i"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])

def chiantipy_ion_tolatex(ions):
    ion_latex = []
    for ion in ions:
        element, ion = ion.split('_')
        ion_latex.append(element[0].upper()+element[1:]+'\,{\sc '+write_roman(int(ion))+'}')
    return ion_latex

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
    sorted_ints = np.char.mod('%.2f', sorted_ints)
    sorted_wvls = np.char.mod('%.2f', sorted_wvls)
    file_path = 'linelist_table.tex'
    ascii.write([chiantipy_ion_tolatex(sorted_ions),sorted_wvls,sorted_ints], file_path,
                names=['Ion', '$\lambda$ (\AA\)', 'Rel. Intensity'], format='aastex', overwrite=True)

    # Strip off first and last two lines for easier formatting in paper
    with open(file_path, "r") as fin:
        lines = fin.read().splitlines(True)

    with open('linelist_table_trim.tex', "w") as fout:
        fout.writelines(lines[3:-2])


