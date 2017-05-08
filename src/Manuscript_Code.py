
# coding: utf-8

# In[1]:

import pandas as pd
get_ipython().magic('pylab inline')


# In[2]:

data_directory = '../data/'


# In[3]:

import os
def try_to_make_directory(f):
    try:
        os.makedirs(f)
    except OSError:
        pass


# Organize data for citations, co-classifications and occurrences
# ===

# In[4]:

print("Organizing Citations")
get_ipython().magic('run -i Organize_Citations.py')


# In[5]:

print("Organizing Classifications")
get_ipython().magic('run -i Organize_Classifications.py')


# In[6]:

print("Organizing Occurrences")
get_ipython().magic('run -i Organize_Occurrences.py')


# Define parameters
# ===

# Define classes and entities to analyze
# ---

# In[7]:

class_systems = ['IPC', 'IPC4', 'USPC']
occurrence_entities = {'Firm': ('occurrences_organized.h5', 'entity_classes_Firm'),
                       'Inventor': ('occurrences_organized.h5', 'entity_classes_Inventor'),
                       'Country': ('occurrences_organized.h5', 'entity_classes_Country'),
                       'PID': ('classifications_organized.h5', 'patent_classes'),
                       }
entity_types = list(occurrence_entities.keys())


# Define what years to calculate networks for
# ---

# In[8]:

# target_years = [2010]
# target_years = 'all'
target_years_dict = {'IPC': 'all',
                    'IPC4': [2010],
                    'USPC': [2010]}


# Define number of years of history networks should include
# ---

# In[9]:

n_years = 'all'

if n_years is None or n_years=='all' or n_years=='cumulative':
    n_years_label = ''
else:
    n_years_label = '%i_years_'%n_years


# In[10]:

citation_metrics = ['Class_Cites_Class_Count',
                    'Class_Cited_by_Class_Count',
                   'Class_Cites_Class_Input_Cosine_Similarity',
                   'Class_Cites_Class_Output_Cosine_Similarity',
                   'Class_Cites_Patent_Input_Cosine_Similarity',
                   'Patent_Cites_Class_Output_Cosine_Similarity',
                   'Class_CoCitation_Count']


# Calculate empirical networks
# ===

# In[8]:

try_to_make_directory(data_directory+'Class_Relatedness_Networks/')
try_to_make_directory(data_directory+'Class_Relatedness_Networks/citations/')
try_to_make_directory(data_directory+'Class_Relatedness_Networks/cooccurrence/')


# In[9]:

### Create empirical networks
randomized_control = False

for class_system in class_systems:
    target_years = target_years_dict[class_system]
    print("Calculating for %s------"%class_system)
    ### Calculate citation networks
    get_ipython().magic('run -i Calculating_Citation_Networks.py')
    all_networks = networks
    
    ### Calculate co-occurrence networks
    preverse_years = True
    for entity_column in entity_types:
        print(entity_column)
        occurrence_data, entity_data = occurrence_entities[entity_column]
        get_ipython().magic('run -i Calculating_CoOccurrence_Networks.py')
        all_networks.ix['Class_CoOccurrence_Count_%s'%entity_column] = networks
    
    ind = ['Class_CoOccurrence_Count_%s'%entity for entity in entity_types]
    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/cooccurrence/class_relatedness_networks_cooccurrence.h5', 
                    mode='a', table=True)
    store.put('/empirical_cooccurrence_%s%s'%(n_years_label,class_system), all_networks.ix[ind], 'table', append=False)
    store.close()
    
    #### Combine them both
    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/class_relatedness_networks.h5', 
                        mode='a', table=True)
    store.put('/empirical_'+n_years_label+class_system, all_networks, 'table', append=False)
    store.close()


# Calculate randomized networks
# ====

# Make directories
# ---

# In[11]:

try_to_make_directory(data_directory+'Class_Relatedness_Networks/citations/controls/')
try_to_make_directory(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/')


# Run randomizations
# ---
# (Currently set up to use a cluster)

# In[12]:

first_rand_id = 0
n_randomizations = 1000
overwrite = True


# In[12]:

python_location = '/home/jeffrey_alstott/anaconda3/bin/python'
from os import path
abs_path_data_directory = path.abspath(data_directory)+'/'


try_to_make_directory('jobfiles/')

for class_system in class_systems:
    target_years = target_years_dict[class_system]
    ### Citations
    try_to_make_directory(data_directory+'Class_Relatedness_Networks/citations/controls/'+class_system)
    basic_program = open('Calculating_Citation_Networks.py', 'r').read()
    job_type = 'citations'
    options="""class_system = %r
target_years = %r
n_years = %r
data_directory = %r
randomized_control = True
citation_metrics = %r
"""%(class_system, target_years, n_years, abs_path_data_directory, citation_metrics)
    
    get_ipython().magic('run -i Calculating_Synthetic_Networks_Control_Commands')

    ### Co-occurrences
    try_to_make_directory(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/'+class_system)
    basic_program = open('Calculating_CoOccurrence_Networks.py', 'r').read()
    job_type = 'cooccurrence'
    for entity in entity_types:
        occurrence_data, entity_data = occurrence_entities[entity]
        options = """class_system = %r
target_years = %r
n_years = %r
data_directory = %r
randomized_control = True
preserve_years = True
chain = False
occurrence_data = %r
entity_data = %r
entity_column = %r
print(occurrence_data)
print(entity_data)
print(entity_column)
"""%(class_system, target_years, n_years, abs_path_data_directory, occurrence_data, entity_data, entity)
    
        get_ipython().magic('run -i Calculating_Synthetic_Networks_Control_Commands')


# Integrate randomized data and calculate Z-scores
# ---
# Note: Any classes that have no data (i.e. no patents within that class) will create z-scores of 'nan', which will be dropped when saved to the HDF5 file. Therefore, the z-scores data will simply not includes these classes.

# In[13]:

n_controls = n_randomizations

output_citations = 'class_relatedness_networks_citations'
output_cooccurrence = 'class_relatedness_networks_cooccurrence'
combine_outputs = True

cooccurrence_base_file_name = 'synthetic_control_cooccurrence_'+n_years_label+'%s_preserve_years_%s'

for class_system in class_systems:
    get_ipython().magic('run -i Calculating_Synthetic_Networks_Integrate_Runs.py')


# Regress out popularity from relatedness measures
# ---
# First create popularity-by-year networks for all class systems and n_years

# In[1]:

# %run -i Calculating_Popularity_Networks.py
# %run -i Regressing_Popularity_Out_of_Z_Scores.py


# Organize individual runs of IPC and store separately
# ---

# In[ ]:

class_system = 'IPC'
target_year = 2010

get_ipython().magic('run -i Calculating_Synthetic_Networks_Organize_Runs.py')


# Delete individual runs of randomizations
# ===

# In[49]:

# from shutil import rmtree

# for class_system in class_systems:
#     if class_system not in ['IPC']:
#         rmtree(data_directory+'Class_Relatedness_Networks/citations/controls/'+class_system)
#         rmtree(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/'+class_system)  


# Make randomized controls of IPC co-occurrence networks without preserving year-by-year structure
# ===

# In[19]:

class_system = 'IPC'
preserve_years = False

first_rand_id = 0
n_randomizations = 1000
overwrite = True

python_location = '/home/jeffrey_alstott/anaconda3/bin/python'
from os import path
abs_path_data_directory = path.abspath(data_directory)+'/'


try_to_make_directory('jobfiles/')

basic_program = open('Calculating_CoOccurrence_Networks.py', 'r').read()
job_type = 'cooccurrence'
for entity in occurrence_entities.keys():
    occurrence_data, entity_data = occurrence_entities[entity]
    options = """class_system = %r
target_years = %r
n_years = %r
data_directory = %r
randomized_control = True
preserve_years = False #This is the important difference
chain = False
occurrence_data = %r
entity_data = %r
entity_column = %r
print(occurrence_data)
print(entity_data)
print(entity_column)
"""%(class_system, target_years, n_years, abs_path_data_directory, occurrence_data, entity_data, entity)
    
    get_ipython().magic('run -i Calculating_Synthetic_Networks_Control_Commands')


# In[4]:

class_system = 'IPC'
output_cooccurrence = 'class_relatedness_networks_cooccurrence_no_preserve_years'
output_citations = False
combine_outputs = False

cooccurrence_base_file_name = 'synthetic_control_cooccurrence_'+n_years_label+'%s_no_preserve_years_%s'

get_ipython().magic('run -i Calculating_Synthetic_Networks_Integrate_Runs.py')


# Make figures
# ===

# In[ ]:

figures_directory = '../manuscript/figs/'
try_to_make_directory(figures_directory)
save_as_manuscript_figures = True

for class_system in class_systems:
    get_ipython().magic('run -i Manuscript_Figures.py')

