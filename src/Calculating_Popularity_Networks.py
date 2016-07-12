
# coding: utf-8

# In[1]:

# import pandas as pd
# %pylab inline


# In[5]:

# data_directory = '../data/'
# class_systems = ['IPC', 'IPC4']
# all_n_years = ['all', 1, 5]

def create_n_years_label(n_years):
    if n_years is None or n_years=='all' or n_years=='cumulative':
        n_years_label = ''
    else:
        n_years_label = '%i_years_'%n_years
    return n_years_label


# In[85]:

store_counts = pd.HDFStore(data_directory+'popularity_counts.h5')
store_networks = pd.HDFStore(data_directory+'popularity_networks.h5')


# In[4]:

all_inventorships = pd.read_csv(data_directory+'disamb_data_ipc_citations_2.csv')
all_inventorships.rename(columns={'IPC3': 'IPC'}, inplace=True)


# In[111]:

popularity_count_of = 'patent'

for class_system in class_systems:
    print(class_system)
    
    data = all_inventorships[['PID',
                          class_system, 'GYEAR']]

    data.rename(columns={'PID': 'patent',
                         class_system: 'Class',
                        'GYEAR': 'Year'},
                inplace=True)

    data.drop_duplicates([popularity_count_of], inplace=True)
    #         class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', '%s_class_lookup'%class_system)
    #         data['Class'] = class_lookup.ix[data['Class']].values
    data.dropna(inplace=True)

    class_size = data.groupby(['Class', 'Year']).count().reset_index().sort('Year').set_index(['Class', 'Year'])
    class_size = class_size.reindex(pd.MultiIndex.from_product([sort(data['Class'].unique()), 
                                                                sort(data['Year'].unique())],
                                                              names=['Class', 'Year'])).fillna(0)
    class_size_cumulative = class_size.groupby(level='Class')[popularity_count_of].cumsum()

#     for n_years in all_n_years:
#         print(n_years)

    if n_years is None or n_years=='all' or n_years=='cumulative':
        this_class_size = class_size_cumulative
    else:
        this_class_size = class_size.groupby(level='Class').apply(lambda x: 
                                                                  pd.rolling_sum(x, n_years))
        this_class_size[popularity_count_of].fillna(class_size_cumulative, inplace=True)

    this_class_size.name = popularity_count_of
    this_class_size = pd.DataFrame(this_class_size.sort_index())

    store_counts['%s_count_%s%s'%(popularity_count_of, 
                            create_n_years_label(n_years), 
                            class_system)] = this_class_size

    patent_count_links = pd.Panel(items=this_class_size.index.levels[1],
                             major_axis=this_class_size.index.levels[0],
                             minor_axis=this_class_size.index.levels[0]
                             )
    for g in this_class_size.groupby(level='Year'):
        patent_count_links.ix[g[0]] = outer(g[1].values, g[1].values)
    store_networks['%s_count_%s%s'%(popularity_count_of, 
                            create_n_years_label(n_years), 
                            class_system)] = patent_count_links


# In[112]:

all_inventorships.sort('GYEAR', inplace=True)

popularity_count_of = 'inventor'

for class_system in class_systems:
    print(class_system)
    data = all_inventorships[['INVENTOR_ID', class_system, 'GYEAR']]

    data.rename(columns={'INVENTOR_ID': 'inventor',
                         class_system: 'Class',
                        'GYEAR': 'Year'},
                inplace=True)

    data.drop_duplicates([popularity_count_of], inplace=True)
    #         class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', '%s_class_lookup'%class_system)
    #         data['Class'] = class_lookup.ix[data['Class']].values
    data.dropna(inplace=True)

    class_size = data.groupby(['Class', 'Year']).count().reset_index().sort('Year').set_index(['Class', 'Year'])
    class_size = class_size.reindex(pd.MultiIndex.from_product([sort(data['Class'].unique()), 
                                                                sort(data['Year'].unique())],
                                                              names=['Class', 'Year'])).fillna(0)
    class_size_cumulative = class_size.groupby(level='Class')[popularity_count_of].cumsum()

#     for n_years in all_n_years:
#         print(n_years)

    if n_years is None or n_years=='all' or n_years=='cumulative':
        this_class_size = class_size_cumulative
    else:
        this_class_size = class_size.groupby(level='Class').apply(lambda x: 
                                                                  pd.rolling_sum(x, n_years))
        this_class_size[popularity_count_of].fillna(class_size_cumulative, inplace=True)

    this_class_size.name = popularity_count_of
    this_class_size = pd.DataFrame(this_class_size.sort_index())

    store_counts['new_%s_count_%s%s'%(popularity_count_of, 
                            create_n_years_label(n_years), 
                            class_system)] = this_class_size

    patent_count_links = pd.Panel(items=this_class_size.index.levels[1],
                             major_axis=this_class_size.index.levels[0],
                             minor_axis=this_class_size.index.levels[0]
                             )
    for g in this_class_size.groupby(level='Year'):
        patent_count_links.ix[g[0]] = outer(g[1].values, g[1].values)
    store_networks['new_%s_count_%s%s'%(popularity_count_of, 
                            create_n_years_label(n_years), 
                            class_system)] = patent_count_links


# In[113]:

store_counts.close()
store_networks.close()

