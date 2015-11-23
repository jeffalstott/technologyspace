
# coding: utf-8

# Setup
# ====

# In[4]:

# import readline  #Import this here so that later rpy2 actually works on the cluster.
import pandas as pd
import seaborn as sns
from pylab import *


# Define What Class System to Analyze
# ===
# Leave commented out to be determined by the Control_Commands notebook when sending to a cluster

# In[2]:

# class_system = 'USPC'
# class_system = 'IPC'
# class_system = 'IPC4'


# In[47]:

# randomized_control = True
# randomized_control = False


# In[48]:

# preserve_years = True


# In[49]:

# chain = 1000
# chain = False


# In[1]:

# occurrence_data = 'occurrences_organized.h5'
# entity_data = 'entity_classes_Firm'
# entity_column = 'Firm'
# entity_data = 'entity_classes_Inventor'
# entity_column = 'Inventor'


# In[51]:

# occurrence_data = 'classifications_organized.h5'
# entity_data = 'patent_classes'
# entity_column = 'PID'


# How Years of History are We Using?
# ===

# In[52]:

# n_years = 1

if n_years is None or n_years=='all' or n_years=='cumulative':
    years_label = ''
else:
    years_label = '%i_years_'%n_years


# What Years are We Calculating Networks for?
# ===

# In[ ]:

# target_years = 'all' #Calculate a network for every year


# Import Occurrence Data
# ===

# In[6]:

# data_directory = '../data/'


# In[7]:

store = pd.HDFStore(data_directory+occurrence_data)

entity_classes = store[entity_data+'_'+class_system].reset_index()

class_lookup = store['%s_class_lookup'%class_system]

store.close()

#Set columns of the patent classification system we're interested in to the default names, without a class system tag.
for column in entity_classes.columns:
    if class_system in column:
        new_name = column.replace('_'+class_system, "")
        entity_classes.rename(columns={column: new_name}, inplace=True)


# In[9]:

#For each kind of entity we have a dataframe with a column name for that entity (i.e. 'Firm', 'Country', and 'Inventor').
#Make a generic column 'Entity' in each of these dataframes with the same information.
entity_classes.rename(columns={entity_column: 'Entity'}, inplace=True)


# In[56]:

### Drop occurrences where data is undefined. This includes if the entity is undefined, such as 
### there being no assignee (e.g. if the patent was assigned to the inventor).
### Class data is also undefined if the patent had been assigned to a class that is not included
### in the set of classes we're analyzing (defined in data/class_lookup_tables.h5)
### In practice this means about 350 patents are removed, which between them have 100 classes
### that aren't represented anywhere else. We don't know if these small or unique classes are
### clerical errors or if they were classes the patent office experimented with creating and
### then dropped; all these classes are not in the current IPC system, so we treat them as noise
### and drop them.

entity_classes.dropna(inplace=True)


# Calculate and Store Class-Class Similarity Metrics for the Empirical Networks and a Bunch of Randomized Controls
# ====

# In[57]:

#classes = sort(list(set(citations['Class_Cited_Patent'].unique()).union(citations['Class_Citing_Patent'].unique())))
classes = arange(len(class_lookup))
years = sort(entity_classes['Year'].unique())
years = list(range(min(years), max(years)+1))
if target_years is None or target_years=='all':
    target_years = years


# In[58]:

def cooccurrence_counts(entity_classes):
    import scipy.sparse
    cooccurrences = scipy.sparse.csr_matrix((ones_like(entity_classes['Entity']),
                                                      (entity_classes['Entity'], 
                                                       entity_classes['Class'])))

    present_cooccurrence = (cooccurrences.T * cooccurrences).todense()
    
    all_cooccurrences = zeros((max(classes)+1, max(classes)+1))
    all_cooccurrences[:present_cooccurrence.shape[0], 
                          :present_cooccurrence.shape[1]] = present_cooccurrence
    
    return all_cooccurrences


# In[59]:

def calculate_cooccurrence_networks(entity_classes,
                                    target_years,
                                    classes=classes,
                                    n_years=n_years
                                   ):
    networks = {}
    for year in target_years:
#         print(year)
        if n_years is None or n_years=='all' or n_years=='cumulative':
            these_entity_classes = entity_classes[entity_classes['Year']<=year]
        else:
            these_entity_classes = entity_classes[((entity_classes['Year']<=year) & 
                                                   (entity_classes['Year']>(year-n_years))
                                                  )]
        networks[year] = cooccurrence_counts(these_entity_classes)
    return pd.Panel(networks)


# In[60]:

if randomized_control:
    import BiRewire as br

    def randomize_occurrences(entity_classes,
                              years=None,
                             preserve_years=True):

#         import rpy2.robjects as ro
#         from rpy2.robjects.packages import importr
#         from rpy2.robjects.numpy2ri import numpy2ri
#         ro.numpy2ri.activate()
#         importr('igraph')
#         importr('BiRewire')


        rewired_entity_classes = pd.DataFrame(columns=['Entity', 'Class', 'Year'],
                                             index=range(len(entity_classes))
                                             )

        if not preserve_years:
            entities, classes = randomize_occurrences_helper(entity_classes[['Entity', 'Class']])#, ro)
            rewired_entity_classes['Entity'] = entities
            rewired_entity_classes['Class'] = classes
            rewired_entity_classes['Year'] = entity_classes['Year']
        else:
            if years is None:
                years = sort(entity_classes['Year'].unique())

            this_start_ind = 0
            for target_year in years:
#                 print(target_year)
                these_entity_classes = entity_classes[entity_classes['Year']==target_year][['Entity', 'Class']]

                entities, classes = randomize_occurrences_helper(these_entity_classes)#, ro)

                n_classifications = entities.shape[0]

                rewired_entity_classes.iloc[this_start_ind:n_classifications+this_start_ind, 0] = entities
                rewired_entity_classes.iloc[this_start_ind:n_classifications+this_start_ind, 1] = classes
                rewired_entity_classes.iloc[this_start_ind:n_classifications+this_start_ind, 2] = target_year

                this_start_ind += n_classifications
        return rewired_entity_classes.astype('int64')
    
    def randomize_occurrences_helper(entity_classes):#,
#                                     ro):
        
        Entity_lookup = pd.Series(index=entity_classes.Entity.unique(),
                                  data=1+arange(entity_classes.Entity.nunique()))
        Class_lookup = pd.Series(index=entity_classes.Class.unique(),
                                 data=1+arange(entity_classes.Class.nunique()))

        n_entities = len(Entity_lookup)
        n_classes = len(Class_lookup)

        entity_classes.Entity = Entity_lookup.ix[entity_classes.Entity].values
        entity_classes.Class = Class_lookup.ix[entity_classes.Class].values
        entity_classes.Class += n_entities

#         entity_classes = entity_classes.values.ravel(order='C')
#         ro.globalenv['entity_classes'] = ro.Vector(entity_classes)
#         ro.globalenv['n_entities'] = ro.default_py2ri(n_entities)
#         ro.globalenv['n_classes'] = ro.default_py2ri(n_classes)    
#         ro.r('g = graph.bipartite(c(rep(T, n_entities), rep(F, n_classes)), entity_classes)')
#         ro.r('h = birewire.rewire.bipartite(g, verbose=FALSE, exact=TRUE)')
#         z = array(ro.r('z = get.edgelist(h)')).astype('int')
        this_rewiring = br.Rewiring(data=entity_classes.values,
                                   type_of_array='edgelist_b',
                                   type_of_graph='bipartite')
        this_rewiring.rewire(verbose=0)   
        z = this_rewiring.data_rewired

        
        Entity_lookup = pd.DataFrame(Entity_lookup).reset_index().set_index(0)
        Class_lookup = pd.DataFrame(Class_lookup).reset_index().set_index(0)
        
        entities = Entity_lookup.ix[z[:,0]].values.flatten()
        classes = Class_lookup.ix[z[:,1]-n_entities].values.flatten()
        
        return entities, classes


# In[61]:

if randomized_control:
    entity_classes = randomize_occurrences(entity_classes,
                                           preserve_years=preserve_years)
networks = calculate_cooccurrence_networks(entity_classes, target_years)


# In[62]:

if randomized_control and chain: #If we have a chained randomization process, then keep going!
    randomizations = {0: networks}
    for iteration in range(1,chain):
        if not iteration%100:
            print(iteration)
        entity_classes = randomize_occurrences(entity_classes, 
                                              preserve_years=preserve_years) 
        randomizations[iteration] = calculate_cooccurrence_networks(entity_classes, target_years)
    networks = pd.Panel4D(randomizations)


# In[63]:

networks.major_axis = class_lookup.index[networks.major_axis]
networks.minor_axis = class_lookup.index[networks.minor_axis]


# In[64]:

if randomized_control:    
    if preserve_years:
        file_name = 'synthetic_control_cooccurrence_%s%s_preserve_years_%s'%(years_label, entity_column, class_system)
    else:
        file_name = 'synthetic_control_cooccurrence_%s%s_no_preserve_years_%s'%(years_label, entity_column, class_system)
    if chain:
        file_name += '_chain.h5'
    else:
        file_name += '_%s.h5'%randomization_id
    
#     file_name = '5_years_'+file_name

    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/cooccurrence/controls/%s/%s'%(class_system,file_name),
                        mode='w', table=True)
    store.put('/synthetic_cooccurrence_%s_%s'%(entity_column, class_system), networks, 'table', append=False)
    store.close()
else:
    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/cooccurrence/class_relatedness_networks_cooccurrence.h5',
                        mode='a', table=True)
#     df = store['empirical_%s'%class_system]
#     df.ix['Class_CoOccurrence_Count_%s'%entity_column] = networks
    df = networks
    store.put('/empirical_cooccurrence_%s%s_%s'%(years_label, entity_column, class_system), df, 'table', append=False)
    store.close()

