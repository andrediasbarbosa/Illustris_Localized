# Step0 Preparing Data
# Step1 Cloning illustris-api locally, running pip install all and then configuring IDE
# Step2 https://stackoverflow.com/questions/31235376/pycharm-doesnt-recognize-installed-module
# Also see PyCharm help at https://www.jetbrains.com/help/pycharm/

#Total number of Halos = 4231400
# Try this: https://www.tng-project.org/data/forum/topic/31/plotting-number-of-halos-per-mass-bin/

import illustris_python as ill
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py

def getCount(listOfElems, cond = None):
    'Returns the count of elements in list that satisfies the given condition'
    if cond:
        count = sum(cond(elem) for elem in listOfElems)
    else:
        count = len(listOfElems)
    return count

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

def nested_histogram():

    # Generate some random data
    data1 = np.random.normal(100, 10, 200)
    data2 = np.random.normal(90, 5, 200)
    data3 = np.random.normal(80, 20, 200)

    # Create the figure and axes objects
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Calculate and plot the histograms
    axs[0].hist(data1, bins=20, color='r')
    axs[1].hist(data2, bins=20, color='g')
    axs[2].hist(data3, bins=20, color='b')

    # Add labels and titles
    axs[0].set_title('Data 1')
    axs[1].set_title('Data 2')
    axs[2].set_title('Data 3')
    fig.suptitle('Nested Histograms')

    # Display the plot
    plt.show()

    return

def mass_conversion(float):
    return float * 1e10 / 0.6774

def test_0():
    fields = ['GroupFirstSub']
    snap = 99
    group_first_sub = ill.groupcat.loadHalos(basePath, snap, fields=fields)
    print("group_first_sub.shape = ", group_first_sub.shape)
    print("group_first_sub[9080] = ", group_first_sub[9080])
    print("group_first_sub[9081] = ", group_first_sub[9081])
    print("group_first_sub[9082] = ", group_first_sub[9082])
    return

def test_1():
    HaloMasses = ill.groupcat.loadHalos(basePath, 99, fields=['GroupMass'])
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
    HaloMasses = ill.groupcat.loadHalos(basePath, 99, fields=['Group_M_Crit200'])
    df2 = pd.DataFrame(HaloMasses)
    df2.columns = ['HaloM_Crit200']
    df2.assign(HaloM_Crit200=lambda x: 0.6774*df2['HaloM_Crit200']/100)
    plt.hist(df2['HaloM_Crit200'], bins=100)
    plt.ylabel('Log Number of FoF Halos')
    plt.xlabel('Log10 (Halo Mass_Crit200)')
    plt.title('Halo [Group_Mass_Crit200] Distribution')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    print(df2.describe())

    count = getCount(HaloMasses, lambda x: x > 0.7 and x < 1.5)
    print("\n the number of FoF satisfying the [0,7;1.5]M condition is:" + str(count))

    return

def test_2validation():
    # Open the HDF5 file in read-only mode
    with h5py.File('D:/IllustrisData/TNG100-1-Dark/adhoc/fof_subhalo_tab_099.Group.Group_M_Crit200.hdf5', 'r') as hdf5_file:
        # Print the names of all the groups in the file
        print(list(hdf5_file.keys()))
        # Get the group with the name 'group_name'
        group = hdf5_file['Group']
        # Print the names of all the datasets in the group
        print(list(group.keys()))
        # Get the dataset with the name 'dataset_name'
        dataset = group['Group_M_Crit200']
        # Print the shape and data type of the dataset
        print(dataset.shape)
        print(dataset.dtype)
        # Read the data from the dataset and store it in a NumPy array
        data = dataset[...]
        M200 = data * 1e10 / 0.6774
        print(data[0])
        print(M200[0])

    #HaloMasses = il.groupcat.loadHalos(basePath, 99, fields=['Group_M_Crit200'])
    df2 = pd.DataFrame(M200)
    df2.columns = ['HaloM_Crit200']
    df2.assign(HaloM_Crit200=lambda x: 1e10*df2['HaloM_Crit200']/0.6774)
    plt.hist(df2['HaloM_Crit200'], bins=100)
    plt.ylabel('Log Number of FoF Halos')
    plt.xlabel('Log10 (Halo Mass_Crit200)')
    plt.title('Halo [Group_Mass_Crit200] Distribution')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    print(df2.describe())

    wcount = np.where((M200 > 0.7e12) & (M200 < 1.5e12))[0]
    print("\n the number of FoF satisfying the [0,7;1.5]M condition is:" + str(len(wcount)))

    return


def get_sample_fof():
    # Open the HDF5 file in read-only mode
    # https://www.tng-project.org/api/Illustris-1/files/groupcat-135/?Group=Group_M_Crit200

    with h5py.File('D:/IllustrisData/TNG100-1-Dark/adhoc/fof_subhalo_tab_099.Group.Group_M_Crit200.hdf5', 'r') as hdf5_file:
        group = hdf5_file['Group']
        dataset = group['Group_M_Crit200']
        data = dataset[...]
        M200 = data * 1e10 / 0.6774

    FoFSampleIndex=[]

    for index,value in enumerate(M200):
        if value > 0.7e12 and value < 1.5e12:
            FoFSampleIndex.append(index)

    return FoFSampleIndex

def test_3():

    # Open the HDF5 file in read-only mode (this is the Halo Structure file with the following structure)
    #['E_s', 'GroupFlag', 'Header', 'M200c', 'M_acc_dyn', 'Mean_vel', 'R0p9', 'a_form', 'c200c', 'f_mass_Cen', 'q', 'q_vel', 's', 's_vel', 'sigma_1D', 'sigma_3D']
    # see: https://www.tng-project.org/data/docs/specifications/#sec5q

    # Step0. Validate the M200c count first

    with h5py.File('D:/IllustrisData/TNG100-1-Dark/postprocessing/catalog_name/halo_structure_099.hdf5', 'r') as hdf5_file:
        # Print the names of all the groups in the file
        #print(list(hdf5_file.keys()))
        # Get the group with the name 'M200c'
        M200c = hdf5_file['M200c']
        #print(M200c)
         # Print the shape and data type of the dataset
        #print(M200c.shape)
        #print(M200c.dtype)
        # Read the data from the dataset and store it in a NumPy array
        data = M200c[...]
        M200cc = [10**number for number in data]

    FoFSampleIndex=[]

    for index,value in enumerate(M200cc):
        if value > 0.7e12 and value < 1.5e12:
            FoFSampleIndex.append(index)

    return FoFSampleIndex

def test_structure():

    #with h5py.File("D:/IllustrisData/TNG100-1-Dark/halo_structure/halo_structure_099.hdf5", "r") as f:
    #with h5py.File("D:/IllustrisData/TNG100-1-Dark/adhoc/fof_subhalo_tab_099.Group.Group_M_Crit200.hdf5", "r") as f:
    #with h5py.File("D:/IllustrisData/TNG100-1-Dark/adhoc/fof_subhalo_tab_099.Group.GroupFirstSub.hdf5", "r") as f:
    with h5py.File("D:\IllustrisData\TNG100-1-Dark\postprocessing\catalog_name\subhalo_matching_to_dark.hdf5", "r") as f:
        # Print the name of the file or group
        print(f.name)

        # Iterate over the attributes of the file or group
        for key, value in f.attrs.items():
            print(f"  Attribute: {key}, Value: {value}")

        # Iterate over the datasets in the file or group
        for key in f.keys():
            dset = f[key]
            print(f"  Dataset: {dset.name}")

        # Iterate over the subgroups in the file or group
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                print(f"  Group: {f[key].name}")
                # Print the name of the subgroup
                print(f[key].name)
                # Iterate over the attributes of the subgroup
                for subkey, subvalue in f[key].attrs.items():
                    print(f"    Attribute: {subkey}, Value: {subvalue}")
                # Iterate over the datasets in the subgroup
                for subkey in f[key].keys():
                    subdset = f[key][subkey]
                    print(f"    Dataset: {subdset.name}")
                # Iterate over the subgroups in the subgroup
                for subkey in f[key].keys():
                    if isinstance(f[key][subkey], h5py.Group):
                        print(f"    Group: {f[key][subkey].name}")

    return

def test_4():

    # run test_2 to get the MW analogues Index (based on FoF Catalog) and then run stats on this dataset
    FoFSampleIndex = get_sample_fof()
    # run test_3 to get the MW analogues Index (based on Halo Catalogue) and then run stats on this dataset
    HCSampleIndex = test_3()
    if FoFSampleIndex == HCSampleIndex:
        print("The FoF-HCatalog arrays of MW analogues are equal and have this many records:" + str(len(FoFSampleIndex)))
    else:
        print("The FoF-HCatalog arrays of MW analogues are not equal.\n")

    #https: // www.tng - project.org / data / docs / specifications /  # sec5q
    # Open the HDF5 file -> Halo Structure
    h5py_file = h5py.File("D:/IllustrisData/TNG100-1-Dark/postprocessing/catalog_name/halo_structure_099.hdf5", "r")
    datasets = list(h5py_file.keys())
    # Create an empty dataframe
    df = pd.DataFrame()
    # Iterate through the datasets and add them as columns to the dataframe
    for dataset in datasets:
        if dataset =='GroupFlag' or dataset =='M200c' or dataset =='E_s' or dataset =='sigma_3D' or dataset =='f_mass_Cen':
            df[dataset] = h5py_file[dataset]
            #need to add only relevant entries [FoFSampleIndex]
    # Close the H5PY file
    h5py_file.close()

    # Open the HDF5 file -> Original FoF Halos
    # https://www.tng-project.org/api/TNG100-1-Dark/files/groupcat-99/?Group=GroupFirstSub
    with h5py.File('D:/IllustrisData/TNG100-1-Dark/adhoc/fof_subhalo_tab_099.Group.GroupFirstSub.hdf5', 'r') as hdf5_a_file:
        print(list(hdf5_a_file.keys()))
        group = hdf5_a_file['Group']
        dataset = group['GroupFirstSub']
        df['GroupFirstSub'] = dataset[...]
    hdf5_a_file.close()
    #print(df.describe())

    #printing out the main features of the first Fof (MW analogue)
    index = FoFSampleIndex[10]

    #print("The " + str(index) + "th Galaxy has the following features:")
    #print("GroupFlag=" + str(df['GroupFlag'][index]))
    #print("GroupFirstSub=" + str(df['GroupFirstSub'][index]))
    #print("M200c=" + str(df['M200c'][index]))
    #print("E_s=" + str(df['E_s'][index]))
    #print("sigma_3D=" + str(df['sigma_3D'][index]))
    #print("f_mass_Cen=" + str(df['f_mass_Cen'][index]))

    return df

def test_subhalo_gasfrac():

    # run test_2 to get the MW analogues Index (based on FoF Catalog) and then run stats on this dataset
    FoFSampleIndex = get_sample_fof()

    # now we need to get the SubHalo indexes for these
    snap = 99
    fields = ['GroupFirstSub']
    halos = ill.groupcat.loadHalos(basePath, snap, fields=fields)
    subHaloSampleIndex = []
    for item in FoFSampleIndex:
        subHaloSampleIndex.append(halos[item])

    print(subHaloSampleIndex)
    ptNumDm = ill.snapshot.partTypeNum('dm') # 1
    ptNumGas = ill.snapshot.partTypeNum('gas')  # 0
    ptNumStars = ill.snapshot.partTypeNum('stars')  # 4

    # retrieving some specific values for each SubHalo
    for item in subHaloSampleIndex[:10]:
        all_fields = ill.groupcat.loadSingle(basePath, 99, subhaloID=item)
        gas_mass = all_fields['SubhaloMassInHalfRadType'][ptNumGas]
        stars_mass = all_fields['SubhaloMassInHalfRadType'][ptNumStars]
        #frac = gas_mass / (gas_mass + stars_mass)
        #print(i, group_first_sub[i], frac)
        print("Subhalo:{} has gas_mass={} and stars_mass={})".format(item, gas_mass, stars_mass))
    return

def baryon_dm_values(val,map):
    return map.index(val)

def disk_values(val,map1,map2):
    return map2[map1.index(val)]

def test_5():
    # Augment the FoFSampleInde with
    # GroupFirstSubBaryonic [revmap->GroupFirstSub]
    # + the following fields from (c) Stellar Circularities:
    # CircAbove07Frac ; CircAbove07MinusBelowNeg07Frac ; CircTwiceBelow0Frac
    # https://www.tng-project.org/data/docs/specifications/#sec5c

    #Step 0 -> build a dataframe with the Mc200 Halo Sample, including [DM GroupFirstSub]
    data = test_4()
    FoFSampleIndex = get_sample_fof()
    Halodata = data.reset_index()
    Halodata = Halodata[Halodata.index.isin(FoFSampleIndex)]
    print(Halodata)

    #Step 1 -> add the Baryonic counterpart to the latter as [DM_Baryon_SubLink]
    mapsubbaryonic_dm = []
    with h5py.File('D:\IllustrisData\TNG100-1-Dark\postprocessing\catalog_name\subhalo_matching_to_dark.hdf5', 'r') as hdf5_a_file:
        print(list(hdf5_a_file.keys()))
        group = hdf5_a_file['Snapshot_99']
        dataset = group['SubhaloIndexDark_LHaloTree']
        mapsubbaryonic_dm = list(dataset)
    hdf5_a_file.close()

    Halodata['Baryon_LsHalo'] = Halodata.apply(lambda row: baryon_dm_values(row['GroupFirstSub'],mapsubbaryonic_dm), axis=1)
    print(Halodata)

    # Step 2 -> add the (d) Stellar Circularities columns
    # [CircAbove07Frac] ; [CircAbove07MinusBelowNeg07Frac] ; [CircTwiceBelow0Frac]
    mapsubfind_id = []
    mapcircabove07 = []
    with h5py.File('D:\IllustrisData\TNG100-1-Dark\postprocessing\catalog_name\stellar_circs.hdf5', 'r') as hdf5_b_file:
        print(list(hdf5_b_file.keys()))
        group = hdf5_b_file['Snapshot_99']
        dataset1 = group['SubfindID']
        dataset2 = group['CircAbove07Frac']
        mapsubfind_id = list(dataset1)
        mapcircabove07 = list(dataset2)
    hdf5_b_file.close()

    Halodata['SubHalo_CircAbove07'] = Halodata.apply(lambda row: disk_values(row['Baryon_LsHalo'],mapsubfind_id,mapcircabove07), axis=1)
    print(Halodata)

    # Step 3 -> Plotting the Histogram of [SubHalo_CircAbove07]
    Halodata['SubHalo_CircAbove07'].hist(bins=200)
    plt.title('Histogram of Stellar Circularities for Sample of MW-like Halos')
    plt.xlabel('SubHalo_CircAbove07')
    plt.ylabel('Frequency')
    plt.show()

    return

def test_6():

    mapcircabove07 = []
    with h5py.File('D:\IllustrisData\TNG100-1-Dark\postprocessing\catalog_name\stellar_circs.hdf5', 'r') as hdf5_b_file:
        print(list(hdf5_b_file.keys()))
        group = hdf5_b_file['Snapshot_99']
        dataset = group['CircAbove07Frac']
        mapcircabove07 = list(dataset)
    hdf5_b_file.close()

    plt.hist(mapcircabove07,bins=200, color='orange')
    plt.title('Histogram of Stellar Circularities for All (sub)Halos with M⋆>3.4×10^8M⊙')
    plt.xlabel('SubHalo_CircAbove07')
    plt.ylabel('Frequency')
    plt.show()

    return

if __name__ == '__main__':

    # Config basepath accordingly (Linux vs Windows)
    basePath = 'D:/IllustrisData/TNG100-1-Dark/output'
    #basePath = '/home/andre/Illustris_Data/TNG100-1-Dark/output'

    #test_0()
    #test_1()
    #test_2()
    #test_2validation()
    #test_3()
    #test_structure()
    #test_4()
    #test_subhalo_gasfrac()
    #test_5()
    #test_6()
"""
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


