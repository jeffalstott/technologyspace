
# coding: utf-8

# In[1]:

import pandas as pd
import seaborn as sns
get_ipython().magic('pylab inline')


# Import and Organize Data
# ===

# In[2]:

all_data = pd.read_csv(data_directory+'disamb_data_ipc_citations_2.csv')
#                                 parse_dates=[7,8])


# In[4]:

all_data = all_data[['PID',
                      'INVENTOR_ID',
                      'ASSIGNEE_ID',
                      'COUNTRY',
                      'IPC3',
                     'IPC4',
                      'GYEAR']]

all_data.rename(columns={'INVENTOR_ID': 'Inventor',
                         'ASSIGNEE_ID': 'Firm',
                         'COUNTRY': 'Country',
                         'IPC3': 'Class_IPC',
                         'IPC4': 'Class_IPC4',                         
                         'GYEAR': 'Year'
                         },
                inplace=True)


all_data.drop_duplicates(inplace=True)


# Clean IPC classes

# In[5]:

# IPC_classes = sort(all_data['Class_IPC'].unique())
# IPC_class_lookup = pd.Series(index=IPC_classes,
#                       data=arange(len(IPC_classes)))
IPC_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'IPC_class_lookup')
all_data['Class_IPC'] = IPC_class_lookup.ix[all_data['Class_IPC']].values


# In[6]:

# IPC4_classes = sort(all_data['Class_IPC4'].unique())
# IPC4_class_lookup = pd.Series(index=IPC4_classes,
#                       data=arange(len(IPC4_classes)))
IPC4_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'IPC4_class_lookup')
all_data['Class_IPC4'] = IPC4_class_lookup.ix[all_data['Class_IPC4']].values


# Import USPC classes
# ===

# In[8]:

USPC_patent_attributes = pd.read_csv(data_directory+'PATENT_US_CLASS_SUBCLASSES_1975_2011.csv',
                               header=None,
                               names=['Patent', 'Class_USPC', 'Subclass_USPC'])

USPC_patent_attributes.drop(['Subclass_USPC'], axis=1, inplace=True)

#Hope that the first class associated with each patent is the "main" class
USPC_patent_attributes.drop_duplicates(["Patent"], inplace=True) 

USPC_patent_attributes.set_index('Patent', inplace=True)
USPC_patent_attributes.ix[:,'Class_USPC'] = USPC_patent_attributes['Class_USPC'].map(lambda x: x if type(x)==int else int(x) if x.isdigit() else nan)
USPC_patent_attributes.dropna(inplace=True)


### Convert the non-contiguous USPC classes to a contiguous numeric system, and store in the conversion in a lookup table
# USPC_classes = sort(USPC_patent_attributes['Class_USPC'].unique())
# USPC_class_lookup = pd.Series(index=USPC_classes,
#                       data=arange(len(USPC_classes)))
USPC_class_lookup = pd.read_hdf(data_directory+'class_lookup_tables.h5', 'USPC_class_lookup')
USPC_patent_attributes['Class_USPC'] = USPC_class_lookup.ix[USPC_patent_attributes['Class_USPC']].values


# In[9]:

all_data['Class_USPC'] = USPC_patent_attributes.ix[all_data['PID']].values


# Make entity lookup tables
# ===

# In[10]:

Inventors = sort(all_data['Inventor'].unique())
Inventor_lookup = pd.Series(index=Inventors,
                      data=arange(len(Inventors)))

all_data['Inventor'] = Inventor_lookup.ix[all_data['Inventor']].values


# In[11]:

Countries = sort(all_data['Country'].unique().astype('str'))
Country_lookup = pd.Series(index=Countries,
                      data=arange(len(Countries)))

all_data['Country'] = Country_lookup.ix[all_data['Country'].astype('str')].values


# In[12]:

Firms = sort(all_data['Firm'].unique())
Firm_lookup = pd.Series(index=Firms,
                      data=arange(len(Firms)))

all_data['Firm'] = Firm_lookup.ix[all_data['Firm']].values


# Drop duplicates to create a table of entries
# ===

# In[13]:

all_data.sort('Year', inplace=True)


# In[14]:

entity_classes = {}
for entity_type in ['Inventor', 'Firm', 'Country']:
    for class_system in ['IPC', 'USPC', 'IPC4']:
        df = all_data[[entity_type, 
                       'Class_'+class_system, 
                       'Year']].drop_duplicates([entity_type, 
                                                 'Class_'+class_system])
        entity_classes[entity_type+'_'+class_system] = df


# Write Data
# ===

# In[18]:

store = pd.HDFStore(data_directory+'occurrences_organized.h5', mode='w', table=True)
store.put('/IPC_class_lookup', IPC_class_lookup, 'table', append=False)
store.put('/IPC4_class_lookup', IPC4_class_lookup, 'table', append=False)
store.put('/USPC_class_lookup', USPC_class_lookup, 'table', append=False)
store.put('/Inventor_lookup', Inventor_lookup, 'table', append=False)
store.put('/Country_lookup', Country_lookup, 'table', append=False)
store.put('/Firm_lookup', Firm_lookup, 'table', append=False)


for k in entity_classes.keys():
    print("Writing %s"%k)
    store.put('/entity_classes_'+k, entity_classes[k], append=False)
store.close()

