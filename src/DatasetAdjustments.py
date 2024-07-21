import pandas as pd
from collections import defaultdict
from scipy.io import arff

bins_limits = [10, 10]


def dataset_prep_by_hyper_parameters(dataset_name, max_bins_size, sample_size=None, max_tot_elements_num=None, binning_style='regular'):
    # get dataset
    if type(dataset_name) == str:
        if 'arff' in dataset_name:
            dataset = arff.loadarff('Datasets/' + dataset_name)
            dataset = pd.DataFrame(dataset[0])
        else:
            dataset = pd.read_csv('Datasets/' + dataset_name + '.csv')
    else:
        dataset = dataset_name
    dataset = dataset.fillna(0)
    dataset.index = range(len(dataset))

    # handle sample
    if sample_size is not None and sample_size <= len(dataset):
        dataset = dataset.sample(sample_size, random_state=1)
        dataset.index = range(sample_size)

    # change '_' char in column names to '-'
    dataset.columns = [c.replace('_', '-') for c in dataset.columns]

    # bin the data (according to max_bins_size)
    if dataset_name == 'adult':
        binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = adult_adjustments(dataset, max_bins_size, max_tot_elements_num, binning_style=binning_style)
    elif dataset_name == 'spotify_all':
        binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = spotify_adjustments(dataset, max_bins_size, max_tot_elements_num)
    elif dataset_name == 'flights':
        binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = flights_adjustments(dataset, max_bins_size, max_tot_elements_num)
    else: # other datasets (isolet)
        binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = alternate_df(dataset, max_bins_size, max_tot_elements_num)
    return binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict


def cals_columns_to_drop(df, max_bins_size, max_tot_elements_num=None):
    total_bins_sum = 0
    cols_to_drop = []
    for col in df:
        col_unique = len(pd.unique(df[col]))
        if col_unique < max_bins_size:
            col_num = col_unique
        else:
            col_num = max_bins_size
        if total_bins_sum + col_num > max_tot_elements_num:
            cols_to_drop.append(col)
        else:
            total_bins_sum += col_num
    return cols_to_drop


def calc_single_elements_support(elements, binned_df):
    elements_support_dict = defaultdict(float)
    for element in elements:
        element_val = element.split('_')[1]
        element_col = element.split('_')[0]
        elements_support_dict[element] = len(binned_df[binned_df[element_col].astype(str) == str(element_val)]) / len(binned_df)
    return elements_support_dict


# used to isolate, but will work for any dataset
def alternate_df(orig_df, bins_max_size=4, max_tot_elements_num=40):
    orig_df = orig_df.dropna()

    # limit the total elements number (according to max_bins_size and max_tot_elements_num)
    if max_tot_elements_num is not None:
        orig_df = orig_df.drop(cals_columns_to_drop(orig_df, bins_max_size, max_tot_elements_num),axis=1)
    else:
        orig_df = orig_df


    binned_df, binary_df = regular_binning(orig_df, bins_max_size)

    # get avg_bins_df
    val_to_bins_dict, avg_bins_df, elements_support_dict = get_dicts_processed_dfs_from_df(binned_df)

    return binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict


def get_dicts_processed_dfs_from_df(binned_df):
    val_to_bins_dict = {}
    avg_bins_df = pd.DataFrame()

    for column in binned_df:
        if type(binned_df[column][0]) == pd.Interval:
            avg_bins_df[column] = binned_df.apply(lambda row: round(row[column].mid, 2), axis=1)
        else:
            avg_bins_df[column] = binned_df[column]

    # get val_to_bins_dict and bins list
    bins = []
    for index, row in binned_df.iterrows():
        for column in binned_df:
            if type(row[column]) == pd.Interval:
                val_to_bins_dict[column + '_' + str(int(round(row[column].mid, 2)))] = str(row[column])
            else:
                val_to_bins_dict[column + '_' + str(row[column])] = str(row[column])
            bins.append(column + '_' + str(row[column]))

    # get elements_support_dict
    bins = set(bins)
    elements_support_dict = calc_single_elements_support(bins, binned_df)
    return val_to_bins_dict, avg_bins_df, elements_support_dict


def regular_binning(orig_df, bins_max_size=4):
    binned_df = pd.DataFrame()
    binary_df = pd.DataFrame()

    for column in orig_df:
        value_range = orig_df[column].nunique()

        if value_range <= bins_limits[0]:
            binned_df[column] = orig_df[column]
        elif type(orig_df[column][0]) != str:
            binned_df[column] = pd.cut(orig_df[column], bins_max_size, duplicates='drop')
        else:
            binned_df[column] = orig_df[column]
        binary_df = pd.concat([binary_df, pd.get_dummies(binned_df[column]).add_prefix(column+'_')], axis=1)
    return binned_df, binary_df


#################################################
################### Adults ######################
#################################################


def adult_adjustments(orig_df, bins_max_size=4, max_tot_elements_num=40, binning_style='regular'):
    orig_df = orig_df[orig_df['occupation'] != '?']
    orig_df = orig_df[orig_df['workclass'] != '?']
    orig_df = orig_df.drop(columns=['education', 'marital-status', 'fnlwgt'])

    binned_df, binary_df = get_adults_binned_df(orig_df, bins_max_size, max_tot_elements_num, binning_style=binning_style)

    val_to_bins_dict, avg_bins_df, elements_support_dict = get_dicts_processed_dfs_from_df(binned_df)
    return binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict


def get_adults_binned_df(orig_df, bins_max_size=4, max_tot_elements_num=40, binning_style='regular'):
    if binning_style == 'regular':
        if max_tot_elements_num is not None:
            orig_df = orig_df.drop(cals_columns_to_drop(orig_df, bins_max_size, max_tot_elements_num), axis=1)
        else:
            orig_df = orig_df
        binned_df, binary_df = regular_binning(orig_df, bins_max_size)
    else:
        binned_df, binary_df = custom_adults_binning(orig_df, bins_max_size)
    binned_df = binned_df.dropna()
    return binned_df, binary_df


def custom_adults_binning(orig_df, bins_max_size=4):
    binned_df = pd.DataFrame()
    binary_df = pd.DataFrame()

    for column in orig_df:
        value_range = orig_df[column].nunique()
        if column == 'age':
            binned_df[column] = pd.cut(orig_df[column], [0,25,35,45,55,65,75,85,90], duplicates='drop')
        elif column == 'hours-per-week':
            binned_df[column] = pd.cut(orig_df[column], [0,20,40,60,100], duplicates='drop')
        elif column == 'capital-gain':
            binned_df[column] = pd.cut(orig_df[column], [-1,10000,100000], duplicates='drop')
        elif column == 'capital-loss':
            binned_df[column] = pd.cut(orig_df[column], [-1,1000,3000], duplicates='drop')
        elif column == 'educational-num':
            binned_df[column] = pd.cut(orig_df[column], [0,8,12,16], duplicates='drop')
        elif value_range <= bins_limits[0]:
            binned_df[column] = orig_df[column]
        elif type(orig_df[column][0]) != str:
            binned_df[column] = pd.qcut(orig_df[column], bins_max_size,  duplicates='drop')
        else:
            binned_df[column] = orig_df[column]
        binary_df = pd.concat([binary_df, pd.get_dummies(binned_df[column]).add_prefix(column+'_')], axis=1)
    return binned_df, binary_df


##################################################
################### Spotify ######################
##################################################


def spotify_adjustments(orig_df, bins_max_size=4, max_tot_elements_num=40):
    spotify_all = orig_df
    spotify_all.columns = [str.replace(col, "_", "-") for col in spotify_all.columns]
    binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = alternate_df(spotify_all, bins_max_size, max_tot_elements_num)
    return binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict


##################################################
################### flights ######################
##################################################

def flights_adjustments(orig_df, bins_max_size=4, max_tot_elements_num=40):
    flights_all = orig_df
    # get numeric columns only
    flights_all['AIR-SYSTEM-DELAY'] = flights_all['AIR-SYSTEM-DELAY'].fillna(0)
    flights_all['SECURITY-DELAY'] = flights_all['SECURITY-DELAY'].fillna(0)
    flights_all['AIRLINE-DELAY'] = flights_all['AIRLINE-DELAY'].fillna(0)
    flights_all['LATE-AIRCRAFT-DELAY'] = flights_all['LATE-AIRCRAFT-DELAY'].fillna(0)
    flights_all['WEATHER-DELAY'] = flights_all['WEATHER-DELAY'].fillna(0)
    flights_all = flights_all.drop(columns=["FLIGHT-NUMBER", 'YEAR'])
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    flights_all.columns = flights_all.columns.str.replace('_', '')
    flights_all_just_nums = flights_all.select_dtypes(include=numerics)
    flights_all_just_nums = flights_all_just_nums.dropna()
    binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = alternate_df(flights_all_just_nums, bins_max_size, max_tot_elements_num)
    return binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict



