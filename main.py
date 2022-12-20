# Step0 Preparing Data
# Step1 Cloning illustris-api locally, running pip install all and then configuring IDE
# Step2 https://stackoverflow.com/questions/31235376/pycharm-doesnt-recognize-installed-module
# Also see PyCharm help at https://www.jetbrains.com/help/pycharm/

#Total number of Halos = 4231400
# Try this: https://www.tng-project.org/data/forum/topic/31/plotting-number-of-halos-per-mass-bin/

import illustris_python as il
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def running_histogram(X, nBins=100, binSize=None, normFac=None):
    """ Create a adaptive histogram of a (x) point set using some number of bins. """
    if binSize is not None:
        nBins = round( (X.max()-X.min()) / binSize )

    bins = np.linspace(X.min(),X.max(), nBins)
    delta = bins[1]-bins[0]

    running_h = []
    bin_centers = []

    for i, bin in enumerate(bins):
        binMax = bin + delta
        w = np.where((X >= bin) & (X < binMax))

        if len(w[0]):
            running_h.append( len(w[0]) )
            bin_centers.append( np.nanmedian(X[w]) )

    if normFac is not None:
        running_h /= normFac

    return bin_centers, running_h

def test_0():
    fields = ['GroupFirstSub']
    snap = 99
    group_first_sub = il.groupcat.loadHalos(basePath, snap, fields=fields)
    print("group_first_sub.shape = ", group_first_sub.shape)
    print("group_first_sub = ", group_first_sub[1])
    return

def test_1():
    HaloMasses = il.groupcat.loadHalos(basePath, 99, fields=['GroupMass'])
    df2 = pd.DataFrame(HaloMasses)
    df2.columns = ['HaloMass']
    df2.assign(LogMass=lambda x: np.log(df2['HaloMass']))
    df2['LogMass'] = df2.apply(lambda row: np.log10(row[0]), axis=1)
    plt.hist(df2['LogMass'], bins=100)
    plt.ylabel('Number of FoF Halos')
    plt.xlabel('Log10 (Halo Mass)')
    plt.title('Halo [GroupMass] Distribution')
    plt.show()
    print(df2.describe())


def test_2():
    HaloMasses = il.groupcat.loadHalos(basePath, 99, fields=['Group_M_Crit200'])
    df2 = pd.DataFrame(HaloMasses)
    df2.columns = ['HaloM_Crit200']
    plt.hist(df2['HaloM_Crit200'], bins=100)
    plt.ylabel('Log Number of FoF Halos')
    plt.xlabel('Log10 (Halo Mass_Crit200)')
    plt.title('Halo [Group_Mass_Crit200] Distribution')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    print(df2.describe())

    return

if __name__ == '__main__':

    basePath = 'D:/IllustrisData/TNG100-1-Dark/output'

    #test_0()
    #test_1()
    test_2()


"""
def test_groupcat_loadHalos_field():
    fields = ['GroupFirstSub']
    snap = 135
    group_first_sub = ill.groupcat.loadHalos(BASE_PATH_ILLUSTRIS_1, snap, fields=fields)
    print("group_first_sub.shape = ", group_first_sub.shape)
    assert_equal(group_first_sub.shape, (7713601,))
    print("group_first_sub = ", group_first_sub)
    assert_true(np.all(group_first_sub[:3] == [0, 16937, 30430]))
    return


def test_groupcat_loadHalos_all_fields():
    snap = 135
    num_fields = 23
    # Illustris-1, snap 135
    cat_shape = (7713601,)
    first_key = 'GroupBHMass'
    all_fields = ill.groupcat.loadHalos(BASE_PATH_ILLUSTRIS_1, snap)
    print("len(all_fields.keys()) = {} (should be {})".format(len(all_fields.keys()), num_fields))
    assert_equal(len(all_fields.keys()), num_fields)
    key = sorted(all_fields.keys())[0]
    print("all_fields.keys()[0] = '{}' (should be '{}')".format(key, first_key))
    assert_equal(key, first_key)
    shape = np.shape(all_fields[key])
    print("np.shape(all_fields[{}]) = {} (should be {})".format(
        key, shape, cat_shape))
    assert_equal(shape, cat_shape)
    return


def test_groupcat_loadHalos_1():
    fields = ['GroupFirstSub']
    snap = 135
    # Construct a path that should not be found: fail
    fail_path = os.path.join(BASE_PATH_ILLUSTRIS_1, 'failure')
    print("path '{}' should not be found".format(fail_path))
    # `OSError` is raised in python3 (but in py3 OSError == IOError), `IOError` in python2
    assert_raises(IOError, ill.groupcat.loadHalos, fail_path, snap, fields=fields)
    return


def test_groupcat_loadHalos_2():
    fields = ['GroupFirstSub']
    snap = 136
    # Construct a path that should not be found: fail
    print("snap '{}' should not be found".format(snap))
    # `OSError` is raised in python3 (but in py3 OSError == IOError), `IOError` in python2
    assert_raises(IOError, ill.groupcat.loadHalos, BASE_PATH_ILLUSTRIS_1, snap, fields=fields)
    return


def test_groupcat_loadHalos_3():
    # This field should not be found
    fields = ['GroupFailSub']
    snap = 100
    # Construct a path that should not be found: fail
    print("fields '{}' should not be found".format(fields))
    assert_raises(Exception, ill.groupcat.loadHalos, BASE_PATH_ILLUSTRIS_1, snap, fields=fields)
    return


# ==========================
# ====    loadSingle    ====
# ==========================


def test_groupcat_loadSingle():
    # Gas fractions for the first 5 subhalos
    gas_frac = [0.0344649, 0.00273708, 0.0223776, 0.0256707, 0.0134044]

    ptNumGas = ill.snapshot.partTypeNum('gas')  # 0
    ptNumStars = ill.snapshot.partTypeNum('stars')  # 4
    for i in range(5):
        # all_fields = ill.groupcat.loadSingle(BASE_PATH_ILLUSTRIS_1, 135, subhaloID=group_first_sub[i])
        all_fields = ill.groupcat.loadSingle(BASE_PATH_ILLUSTRIS_1, 135, subhaloID=i)
        gas_mass   = all_fields['SubhaloMassInHalfRadType'][ptNumGas]
        stars_mass = all_fields['SubhaloMassInHalfRadType'][ptNumStars]
        frac = gas_mass / (gas_mass + stars_mass)
        # print(i, group_first_sub[i], frac)
        print("subhalo {} with gas frac '{}' (should be '{}')".format(i, frac, gas_frac[i]))
        assert_true(np.isclose(frac, gas_frac[i]))

    return


# ============================
# ====    loadSubhalos    ====
# ============================


def test_groupcat_loadSubhalos():
    fields = ['SubhaloMass', 'SubhaloSFRinRad']
    snap = 135
    subhalos = ill.groupcat.loadSubhalos(BASE_PATH_ILLUSTRIS_1, snap, fields=fields)
    print("subhalos['SubhaloMass'] = ", subhalos['SubhaloMass'].shape)
    assert_true(subhalos['SubhaloMass'].shape == (4366546,))
    print("subhalos['SubhaloMass'] = ", subhalos['SubhaloMass'])
    assert_true(
        np.allclose(subhalos['SubhaloMass'][:3], [2.21748203e+04, 2.21866333e+03, 5.73408325e+02]))

    return
"""


