
# coding: utf-8

# In[1]:

import pandas as pd
from pylab import *


# In[17]:

# class_system = 'IPC4'
# n_controls = 1000


# In[ ]:

# n_years = 'cumulative'
# if n_years is None or n_years=='all' or n_years=='cumulative':
#     n_years_label = ''
# else:
#     n_years_label = '%i_years_'%n_years


# In[ ]:

# output_citations = 'class_relatedness_networks_citations'
# output_cooccurrence = 'class_relatedness_networks_cooccurrence'
# combine_outputs = True

# cooccurrence_base_file_name = 'synthetic_control_cooccurrence_'+n_years_label+'%s_preserve_years_%s'
citations_base_file_name = 'synthetic_control_citations_'+n_years_label+'%s'


# In[3]:

# data_directory = '../data/'

citations_controls_directory = data_directory+'Class_Relatedness_Networks/citations/controls/%s/'%class_system
coocurrence_controls_directory = data_directory+'Class_Relatedness_Networks/cooccurrence/controls/%s/'%class_system


# In[5]:

import gc
from time import time


# In[18]:

def running_stats(df_name,
                  file_name,
                  controls_directory=citations_controls_directory,
                  n_controls=n_controls,
                 ):
    M = None
    all_max = None
    all_min = None
    t = time()
    for randomization_id in range(n_controls):

        if not randomization_id%100:
            print(randomization_id)
            print("%.0f seconds"%(time()-t))
            t = time()
        
        f = '%s_%i.h5'%(file_name, randomization_id)
        try:
            x = pd.read_hdf(controls_directory+f, df_name)
        except:
            print("Data not loading for %s. Continuing."%f)
            continue
            

        if M is None:
            M = x
            S = 0
            all_max = M
            all_min = M
            continue
        k = randomization_id+1
        M_previous = M
        M = M_previous.add( x.subtract(M_previous)/k )
        S = ( x.subtract(M_previous).multiply( x.subtract(M) ) ).add(S)
        all_max = maximum(all_max, M)
        all_min = minimum(all_min, M)
        gc.collect()  
    standard_deviation = sqrt(S/(k-1))

    return M, standard_deviation, all_max, all_min


# In[ ]:

if output_citations:
    M, standard_deviation, all_max, all_min = running_stats('synthetic_citations_%s'%class_system,
                                      citations_base_file_name%class_system,
                                      citations_controls_directory
                                     )

    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/citations/%s.h5'%(output_citations),
                        mode='a', table=True)
    store.put('/randomized_mean_%s%s'%(n_years_label, class_system), M, 'table', append=False)
    store.put('/randomized_std_%s%s'%(n_years_label, class_system), standard_deviation, 'table', append=False)

    store.put('/randomized_max_%s%s'%(n_years_label, class_system), all_max, 'table', append=False)
    store.put('/randomized_min_%s%s'%(n_years_label, class_system), all_min, 'table', append=False)

    z_scores = store['empirical_citations_%s%s'%(n_years_label, class_system)].ix[M.labels].subtract(M).divide(standard_deviation)

    z_scores.values[where(z_scores==inf)]=nan 
    #All the cases where the z-scores are inf is where the 1,000 randomized controls said there should be 0 deviation, BUT
    #the empirical case was different anyway. In each of these cases, the empirical case was JUST slightly off. Sometimes
    #a floating point error, and sometimes off by 1 (the minimal amount for citation counts). We shall treat this as not actually
    #deviating, and so it becomes 0/0, which is equal to nan.

    store.put('/empirical_citations_z_scores_%s%s'%(n_years_label, class_system), z_scores, 'table', append=False)

    store.close()


# In[7]:

if output_cooccurrence:
    M = None
    for entity in ['Firm', 'Country', 'Inventor', 'PID']:
        (M_entity, 
         standard_deviation_entity, 
         all_max_entity, 
         all_min_entity) = running_stats('synthetic_cooccurrence_%s_%s'%(entity, class_system),
                                          cooccurrence_base_file_name%(entity, class_system),
                                          coocurrence_controls_directory
                                         )
        if M is None:
            M = pd.Panel4D({'Class_CoOccurrence_Count_%s'%entity: M_entity})
            standard_deviation = pd.Panel4D({'Class_CoOccurrence_Count_%s'%entity: standard_deviation_entity})
            all_max = pd.Panel4D({'Class_CoOccurrence_Count_%s'%entity: all_max_entity}) 
            all_min = pd.Panel4D({'Class_CoOccurrence_Count_%s'%entity: all_min_entity})
        else:
            M['Class_CoOccurrence_Count_%s'%entity] = M_entity
            standard_deviation['Class_CoOccurrence_Count_%s'%entity] = standard_deviation_entity
            all_max['Class_CoOccurrence_Count_%s'%entity] = all_max_entity
            all_min['Class_CoOccurrence_Count_%s'%entity] = all_min_entity

    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/cooccurrence/%s.h5'%(output_cooccurrence),
                        mode='a', table=True)
    store.put('/randomized_mean_%s%s'%(n_years_label, class_system), M, 'table', append=False)
    store.put('/randomized_std_%s%s'%(n_years_label, class_system), standard_deviation, 'table', append=False)

    store.put('/randomized_max_%s%s'%(n_years_label, class_system), all_max, 'table', append=False)
    store.put('/randomized_min_%s%s'%(n_years_label, class_system), all_min, 'table', append=False)

    try:
        z_scores = store['empirical_cooccurrence_%s%s'%(n_years_label, class_system)].ix[M.labels].subtract(M).divide(standard_deviation)

        z_scores.values[where(z_scores==inf)]=nan 
        #All the cases where the z-scores are inf is where the 1,000 randomized controls said there should be 0 deviation, BUT
        #the empirical case was different anyway. In each of these cases, the empirical case was JUST slightly off. Sometimes
        #a floating point error, and sometimes off by 1 (the minimal amount for citation counts). We shall treat this as not actually
        #deviating, and so it becomes 0/0, which is equal to nan.

        store.put('/empirical_cooccurrence_z_scores_%s%s'%(n_years_label, class_system), z_scores, 'table', append=False)
    except KeyError:
        print("No empirical data saved to calculate z-scores with")
        pass
        
    store.close()


# In[12]:

if combine_outputs:
    
    citation_store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/citations/class_relatedness_networks_citations.h5')
    cooccurrence_store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/cooccurrence/class_relatedness_networks_cooccurrence.h5')
    
    M = citation_store['/randomized_mean_%s%s'%(n_years_label, class_system)]
    standard_deviation = citation_store['/randomized_std_%s%s'%(n_years_label, class_system)]
    all_max = citation_store['/randomized_max_%s%s'%(n_years_label, class_system)]
    all_min = citation_store['/randomized_max_%s%s'%(n_years_label, class_system)]
    z_scores = citation_store['/empirical_citations_z_scores_%s%s'%(n_years_label, class_system)]
    
    M_c = cooccurrence_store['/randomized_mean_%s%s'%(n_years_label, class_system)]
    standard_deviation_c = cooccurrence_store['/randomized_std_%s%s'%(n_years_label, class_system)]
    all_max_c = cooccurrence_store['/randomized_max_%s%s'%(n_years_label, class_system)]
    all_min_c = cooccurrence_store['/randomized_max_%s%s'%(n_years_label, class_system)]
    z_scores_c = cooccurrence_store['/empirical_cooccurrence_z_scores_%s%s'%(n_years_label, class_system)]
    
    for label in M_c.labels:
        M[label] = M_c[label]
        standard_deviation[label] = standard_deviation_c[label]
        all_max[label] = all_max_c[label]
        all_min[label] = all_min_c[label]
        z_scores[label] = z_scores_c[label]
        
    
    combine_store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/class_relatedness_networks.h5', 
                                mode='a', table=True)
   
    combine_store.put('/randomized_mean_%s%s'%(n_years_label, class_system), M, 'table', append=False)
    combine_store.put('/randomized_std_%s%s'%(n_years_label, class_system), standard_deviation, 'table', append=False)

    combine_store.put('/randomized_max_%s%s'%(n_years_label, class_system), all_max, 'table', append=False)
    combine_store.put('/randomized_min_%s%s'%(n_years_label, class_system), all_min, 'table', append=False)

    combine_store.put('/empirical_z_scores_%s%s'%(n_years_label, class_system), z_scores, 'table', append=False)

    combine_store.close()

