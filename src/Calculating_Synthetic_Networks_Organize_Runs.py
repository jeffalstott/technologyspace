
# coding: utf-8

# In[2]:

import pandas as pd
from pylab import *


# In[3]:

# class_system = 'IPC'
# n_controls = 1000
# target_year = 2010


# In[4]:

# n_years = 'cumulative'
if n_years is None or n_years=='all' or n_years=='cumulative':
    n_years_label = ''
else:
    n_years_label = '%i_years_'%n_years


# In[5]:

# occurrence_entities = {'Firm': ('occurrences_organized.h5', 'entity_classes_Firm'),
#                        'Inventor': ('occurrences_organized.h5', 'entity_classes_Inventor'),
#                        'Country': ('occurrences_organized.h5', 'entity_classes_Country'),
#                        'PID': ('classifications_organized.h5', 'patent_classes'),
#                        }
# entity_types = list(occurrence_entities.keys())


# In[6]:

# cooccurrence_base_file_name = 'synthetic_control_cooccurrence_'+n_years_label+'%s_preserve_years_%s'
citations_base_file_name = 'synthetic_control_citations_'+n_years_label+'%s'


# In[7]:

# data_directory = '../data/'

citations_controls_directory = data_directory+'Class_Relatedness_Networks/citations/controls/%s/'%class_system
coocurrence_controls_directory = data_directory+'Class_Relatedness_Networks/cooccurrence/controls/%s/'%class_system


# In[8]:

import gc
from time import time


# In[9]:

def organize_runs(df_name,
                  file_name,
                  controls_directory=citations_controls_directory,
                  n_controls=n_controls,
                  target_year=target_year,
                  controls=None,
                  multiple_metrics=True,
                  target_metric=None
                 ):    
    t = time()
    for randomization_id in range(n_controls):

        if not randomization_id%100:
            print(randomization_id)
            print("%.0f seconds"%(time()-t))
            t = time()
        
        f = '%s_%i.h5'%(file_name, randomization_id)
        try:
            if multiple_metrics:
                x = pd.read_hdf(controls_directory+f, df_name).ix[:,target_year]
            else:
                x = pd.read_hdf(controls_directory+f, df_name).ix[target_year]
        except:
            print("Data not loading for %s. Continuing."%f)
            continue
            

        if controls is None:
            controls = pd.Panel4D(labels=x.items, items=arange(n_controls),
                                  major_axis=x.major_axis, minor_axis=x.minor_axis)
        if multiple_metrics:
            controls.ix[x.items, randomization_id] = x.values
        else:
            controls.ix[target_metric, randomization_id] = x
            
        gc.collect()  

    return controls


# In[10]:

controls = organize_runs('synthetic_citations_%s'%class_system,
                         citations_base_file_name%class_system,
                         citations_controls_directory,
                         controls=None
                        )


# In[18]:

for entity in entity_types:
    controls = organize_runs('synthetic_cooccurrence_%s_%s'%(entity, class_system),
                             cooccurrence_base_file_name%(entity, class_system),
                             coocurrence_controls_directory,
                             controls=controls,
                             multiple_metrics=False,
                             target_metric='Class_CoOccurrence_Count_%s'%entity)
        


# In[20]:

store.close()


# In[21]:

store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/class_relatedness_networks_controls_organized_%s.h5'%class_system,
                   mode='a', table=True)
store.put('/controls_%s'%class_system, controls, 'table', append=False)
store.close()

