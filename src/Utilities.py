from matplotlib import pyplot as plt
from .SharqUtils import *

MAX_TOTAL_ELEMENTS_NUM = 40


def create_single_rules_set(dataset, max_bins_size, max_tot_elements_num, sample_size, score_func, min_support, lift_threshold):
    binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = dataset_prep_by_hyper_parameters(dataset, max_bins_size=max_bins_size, max_tot_elements_num=max_tot_elements_num, sample_size=sample_size, binning_style='custom')
    rules_set = rules_prep_by_hyper_parameters(binary_df, score_func, min_support=min_support, lift_threshold=lift_threshold)
    file_name = 'example_' + dataset
    rules_set.to_csv('Rules/' + file_name + '.csv', index=False)
    print(file_name)


def create_upload_rules_set(datasets):
    max_bins_size = [3, 4, 5]
    sample_size = [2000, 5000, 10000]
    min_support = [0.05, 0.1, 0.15, 0.2]
    lift_limit = [0.05, 0.1, 0.15, 0.2]

    def_max_bins_size = 4
    def_max_tot_elements_num = MAX_TOTAL_ELEMENTS_NUM
    def_sample_size = 10000
    def_score_func = is_score_function
    def_min_support = 0.05
    def_lift_limit = 0.05

    for d in datasets:
        for i in max_bins_size:
            binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = dataset_prep_by_hyper_parameters(d, i, max_tot_elements_num=def_max_tot_elements_num, sample_size=def_sample_size)
            rules_set = rules_prep_by_hyper_parameters(binary_df, def_score_func, def_min_support, def_lift_limit)
            print(len(rules_set))
            print(len(binary_df.columns))
            if len(rules_set) == 0:
                continue
            file_name = d + '_' + 'max_bins_size' + '_' + str(i)
            rules_set.to_csv('Rules/' + file_name + '.csv', index=False)
            print(file_name)

        for i in sample_size:
            binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = dataset_prep_by_hyper_parameters(d, def_max_bins_size, max_tot_elements_num=def_max_tot_elements_num, sample_size=i)
            rules_set = rules_prep_by_hyper_parameters(binary_df, def_score_func, def_min_support, def_lift_limit)

            file_name = d + '_' + 'sample_size' + '_' + str(i) + '.csv'
            rules_set.to_csv('Rules/' + file_name + '.csv', index=False)
            print(len(rules_set))
            print(len(binary_df.columns))
            print(file_name)

        for i in min_support:
            binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = dataset_prep_by_hyper_parameters(d, def_max_bins_size, max_tot_elements_num=def_max_tot_elements_num, sample_size=def_sample_size)
            rules_set = rules_prep_by_hyper_parameters(binary_df, def_score_func, i, def_lift_limit)

            file_name = d + '_' + 'min_support' + '_' + str(i)
            rules_set.to_csv('Rules/' + file_name + '.csv', index=False)
            print(len(rules_set))
            print(len(binary_df.columns))
            print(file_name)

        for i in lift_limit:
            binary_df, val_to_bins_dict, avg_bins_df, elements_support_dict = dataset_prep_by_hyper_parameters(d, def_max_bins_size, max_tot_elements_num=def_max_tot_elements_num, sample_size=def_sample_size)
            rules_set = rules_prep_by_hyper_parameters(binary_df, def_score_func, def_min_support, i)

            file_name = d + '_' + 'lift_limit' + '_' + str(i)
            rules_set.to_csv('Rules/' + file_name + '.csv', index=False)
            print(len(rules_set))
            print(len(binary_df.columns))
            print(file_name)


def rules_prep_by_hyper_parameters(binary_df, score_func=is_score_function, min_support=0.05, lift_threshold=0.05):
    transactions = create_transactions(binary_df)
    rules_set = create_rules_from_transactions(transactions=transactions, lift_threshold=lift_threshold, score_func=score_func, min_support=min_support)
    return rules_set


def get_sharq_visualization(sharq_scores, elems_to_filter=None, fig_name=None, stats_df=None):
    df = pd.DataFrame(columns=['Element', 'SHARQ score'])
    df['Element'] = sharq_scores.keys()
    df['SHARQ score'] = sharq_scores.values()

    df = edit_elements_col(df)

    # sort the df by the SHARQ score from high to low
    df = df.sort_values(by='SHARQ score', ascending=True)

    if elems_to_filter is not None:
        df = df[~df['Element'].isin(elems_to_filter)]

    # change the strings in the 'Element' column to instead of '_' to ', ' and wrap the string in brackets
    df['Element'] = df['Element'].apply(lambda x: '(' + x.replace('_', ', ') + ')')

    for index, row in df.iterrows():
        if row['SHARQ score'] < 0:
            df.at[index, 'Color'] = '#808080'
        else:
            df.at[index, 'Color'] = '#3cb371'
    ax = df.plot.barh(x='Element', y='SHARQ score', edgecolor="black", color=list(df['Color']), legend=None)
    ax.grid(color='silver', linestyle='--', linewidth=0.2)
    y_axis = ax.axes.get_yaxis()
    y_axis.set_label_text('')
    ax.set_axisbelow(True)
    plt.axvline(color='dimgrey')
    plt.margins(0.2, 0.2)
    plt.tick_params(axis='both', which='both', top=False)
    plt.show()


def edit_elements_col(df):
    df['Element'] = df['Element'].apply(lambda x: x.replace('-1', '0'))
    df['Element'] = df['Element'].apply(lambda x: x.replace('[', ''))
    df['Element'] = df['Element'].apply(lambda x: x.replace('(', ''))
    df['Element'] = df['Element'].apply(lambda x: x.replace(']', ''))
    df['Element'] = df['Element'].apply(lambda x: x.replace(', ', ' - '))
    df['Element'] = df['Element'].apply(lambda x: x.replace('1000]', '1K]'))
    df['Element'] = df['Element'].apply(lambda x: x.replace('10000', '10K'))
    df['Element'] = df['Element'].apply(lambda x: x.replace('_', ', '))

    return df

