
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sns
get_ipython().magic('pylab inline')


# Import Years
# ===

# In[2]:

# data_directory = '../data/'


# In[2]:

df = pd.read_csv(data_directory+'pid_issdate_ipc.csv',
                          index_col=0)


# In[3]:

df['Year'] = df.ISSDATE.map(lambda x: int(x[-4:]))
df.drop(['ISSDATE', 'IPC3'], axis=1, inplace=True)
patent_years = df


# Import IPC classes
# ===

# In[13]:

patent_classes_IPC = pd.read_csv(data_directory+'pnts_multiple_ipcs_76_06_valid_ipc.csv')
patent_classes_IPC.rename(columns={' MAINCLS': "Class_IPC"},
                  inplace=True)

# patent_classes_IPC.ix[:,'Class_IPC'] = patent_classes_IPC['Class_IPC'].map(lambda x: x if type(x)==int else int(x) if x.isdigit() else nan)
# patent_classes_IPC.dropna(inplace=True)


# In[15]:

patent_classes_IPC = patent_years.merge(patent_classes_IPC[['PID', 'Class_IPC']],right_on='PID',left_index=True).set_index('PID')
patent_classes_IPC = patent_classes_IPC.reset_index().drop_duplicates().set_index('PID')


# In[17]:

# IPC_classes = sort(patent_classes_IPC['Class_IPC'].unique())
# IPC_class_lookup = pd.Series(index=IPC_classes,
#                       data=arange(len(IPC_classes)))
IPC_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'IPC_class_lookup')
patent_classes_IPC['Class_IPC'] = IPC_class_lookup.ix[patent_classes_IPC['Class_IPC']].values


# Import IPC4 classes
# ===

# In[48]:

patent_classes_IPC4 = pd.read_csv(data_directory+'pnts_multiple_ipcs_76_06_valid_ipc.csv')
patent_classes_IPC4.rename(columns={' SUBCLS': "Class_IPC4"},
                  inplace=True)

# patent_classes_IPC.ix[:,'Class_IPC'] = patent_classes_IPC['Class_IPC'].map(lambda x: x if type(x)==int else int(x) if x.isdigit() else nan)
# patent_classes_IPC.dropna(inplace=True)


# In[41]:

patent_classes_IPC4 = patent_years.merge(patent_classes_IPC4[['PID', 'Class_IPC4']],right_on='PID',left_index=True).set_index('PID')
patent_classes_IPC4 = patent_classes_IPC4.reset_index().drop_duplicates().set_index('PID')


# In[27]:

# IPC4_classes = sort(patent_classes_IPC4['Class_IPC4'].unique())
# IPC4_class_lookup = pd.Series(index=IPC4_classes,
#                       data=arange(len(IPC4_classes)))
IPC4_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'IPC4_class_lookup')

patent_classes_IPC4['Class_IPC4'] = IPC4_class_lookup.ix[patent_classes_IPC4['Class_IPC4']].values


# Import USPC classes
# ====

# In[21]:

patent_classes_USPC = pd.read_csv(data_directory+'PATENT_US_CLASS_SUBCLASSES_1975_2011.csv',
                               header=None,
                               names=['PID', 'Class_USPC', 'Subclass_USPC'])

patent_classes_USPC.ix[:,'Class_USPC'] = patent_classes_USPC['Class_USPC'].map(lambda x: x if type(x)==int else int(x) if x.isdigit() else nan)
patent_classes_USPC.dropna(inplace=True)


# In[22]:

patent_classes_USPC = patent_years.merge(patent_classes_USPC[['PID', 'Class_USPC']],right_on='PID',left_index=True).set_index('PID')
patent_classes_USPC = patent_classes_USPC.reset_index().drop_duplicates().set_index('PID')


# In[23]:

# USPC_classes = sort(patent_classes_USPC['Class_USPC'].unique())
# USPC_class_lookup = pd.Series(index=USPC_classes,
#                       data=arange(len(USPC_classes)))

USPC_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'USPC_class_lookup')
patent_classes_USPC['Class_USPC'] = USPC_class_lookup.ix[patent_classes_USPC['Class_USPC']].values


# Write Data
# ===

# In[35]:

store = pd.HDFStore(data_directory+'classifications_organized.h5', mode='a', table=True)


# In[ ]:

store.put('/IPC_class_lookup', IPC_class_lookup, 'table', append=False)
store.put('/patent_classes_IPC', patent_classes_IPC, 'table', append=False)


# In[37]:

store.put('/IPC4_class_lookup', IPC4_class_lookup, 'table', append=False)
store.put('/patent_classes_IPC4', patent_classes_IPC4, 'table', append=False)


# In[ ]:

store.put('/USPC_class_lookup', USPC_class_lookup, 'table', append=False)
store.put('/patent_classes_USPC', patent_classes_USPC, 'table', append=False)


# In[38]:

store.close()

