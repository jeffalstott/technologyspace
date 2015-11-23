
# coding: utf-8

# Setup
# ====

# In[1]:

# randomization_id = 99999


# In[2]:

from time import time
t = time()


# In[1]:

import pandas as pd
# import seaborn as sns
from pylab import *


# In[ ]:

# data_directory = '/home/jeffrey_alstott/technoinnovation/Data/'


# Parameters
# ===
# Define What Class System to Analyze
# ----

# In[4]:

# class_system = 'USPC'
# class_system = 'IPC'
# class_system = 'IPC4'


# Are We Making a Randomized Control?
# ---

# In[5]:

# randomized_control = False


# What Years are We Calculating Networks for?
# ----

# In[ ]:

# target_years = 'all' #Calculate a network for every year


# How Years of History are We Using?
# ---

# In[6]:

# n_years = None
# n_years = 5

if n_years is None or n_years=='all' or n_years=='cumulative':
    years_label = ''
else:
    years_label = '%i_years_'%n_years


# What Metrics are We Calculating?
# ----

# In[ ]:

# citation_metrics = ['Class_Cites_Class_Count',
#            'Class_Cited_by_Class_Count',
#            'Class_Cites_Class_Input_Cosine_Similarity',
#            'Class_Cites_Class_Output_Cosine_Similarity',
#            'Class_Cites_Patent_Input_Cosine_Similarity',
#            'Patent_Cites_Class_Output_Cosine_Similarity',
# #            'Class_Cites_Patent_Input_Jaccard_Similarity',
# #            'Patent_Cites_Class_Output_Jaccard_Similarity',
#            'Class_CoCitation_Count']


# Import Citation Data
# ===

# In[7]:

citations = pd.read_hdf(data_directory+'citations_organized.h5', 'citations')


class_lookup = pd.read_hdf(data_directory+'citations_organized.h5', 
                           '%s_class_lookup'%class_system)

#Set columns of the patent classification system we're interested in to the default names, without a class system tag.
for column in citations.columns:
    if class_system in column:
        new_name = column.replace('_'+class_system, "")
        citations.rename(columns={column: new_name}, inplace=True)
        
citations = citations[['Citing_Patent', 'Cited_Patent', 
                      'Year_Citing_Patent', 'Class_Citing_Patent',
                      'Year_Cited_Patent', 'Class_Cited_Patent',
                      'Same_Class']]


# In[ ]:

### Drop citations where one of the patents has an undefined class. 
### This would happen if the patent had been assigned to a class that is not included
### in the set of classes we're analyzing (defined in data/class_lookup_tables.h5)
### In practice this means about 350 patents are removed, which between them have 100 classes
### that aren't represented anywhere else. We don't know if these small or unique classes are
### clerical errors or if they were classes the patent office experimented with creating and
### then dropped; all these classes are not in the current IPC system, so we treat them as noise
### and drop them.
citations.dropna(subset=['Class_Citing_Patent', 'Class_Cited_Patent'], inplace=True)


# In[8]:

if randomized_control:
    patent_attributes = pd.read_hdf(data_directory+'citations_organized.h5', 'patent_attributes')
    
    for column in patent_attributes.columns:
        if class_system in column:
            new_name = column.replace('_'+class_system, "")
            patent_attributes.rename(columns={column: new_name}, inplace=True)

    patent_attributes = patent_attributes[['Year', 'Class']]


# Calculate and Store Class-Class Similarity Metrics
# ====

# In[9]:

from time import time
def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print('%2.2f sec' % (te-ts))
        return result
    return timed   


# In[10]:

# if randomized_control:
#     ### Define functions to generate random controls
#     @timeit
#     def randomize_citations(citations,
#                             patent_attributes=patent_attributes):
#         citations_randomized = citations.copy()

#         ### Take the same-class citations of every class and permute them.
#         same_class_ind = citations_randomized['Same_Class']==True
#         citations_randomized.ix[same_class_ind, 'Cited_Patent'] = citations_randomized.ix[same_class_ind].groupby(['Year_Citing_Patent', 
#             'Year_Cited_Patent', 
#             'Class_Citing_Patent', 
#             ])['Cited_Patent'].transform(permutation)

#         ### Take the cross-class citations and permute them.
#         cross_class_ind = -same_class_ind
#         citations_randomized.ix[cross_class_ind, 'Cited_Patent'] = citations_randomized.ix[cross_class_ind].groupby(['Year_Citing_Patent', 
#             'Year_Cited_Patent', 
#             ])['Cited_Patent'].transform(permutation)

#         ### Drop patent attributes (which are now inaccurate for the cited patent) and bring them in from patent_attributes
#         citations_randomized = citations_randomized[['Citing_Patent', 'Cited_Patent', 'Same_Class']]

#         citations_randomized = citations_randomized.merge(patent_attributes, 
#                         left_on='Citing_Patent', 
#                         right_index=True,
#                         )

#         citations_randomized = citations_randomized.merge(patent_attributes, 
#                         left_on='Cited_Patent', 
#                         right_index=True,
#                         suffixes=('_Citing_Patent','_Cited_Patent'))
#         return citations_randomized


# In[11]:

if randomized_control:
    import BiRewire as br
#     import rpy2.robjects as ro
#     from rpy2.robjects.packages import importr
#     from rpy2.robjects.numpy2ri import numpy2ri
#     ro.numpy2ri.activate()
#     importr('igraph')
#     importr('BiRewire')
    
    ### Define functions to generate random controls
    @timeit
    def randomize_citations(citations,
                            patent_attributes=patent_attributes):
        citations_randomized = citations.copy()

        ### Take the same-class citations of every class and permute them.
#         from time import sleep
#         sleep(rand()*10)
        print("Randomizing Same-Class Citations")
        same_class_ind = citations_randomized['Same_Class']==True
        grouper = citations_randomized.ix[same_class_ind].groupby(['Year_Citing_Patent', 
                                                                   'Year_Cited_Patent', 
                                                                   'Class_Citing_Patent', 
                                                                  ])[['Citing_Patent', 
                                                                      'Cited_Patent']]
        print("%i groups"%(len(grouper)))

        citations_randomized.ix[same_class_ind, ['Citing_Patent', 
                                                 'Cited_Patent']
                                ] = grouper.apply(randomize_citations_helper)

        ### Take the cross-class citations and permute them.
#         from time import sleep
#         sleep(rand()*10)
        print("Randomizing Cross-Class Citations")        
        cross_class_ind = -same_class_ind
        grouper = citations_randomized.ix[cross_class_ind].groupby(['Year_Citing_Patent', 
                                                                   'Year_Cited_Patent', 
                                                                  ])[['Citing_Patent', 
                                                                      'Cited_Patent']]
        print("%i groups"%(len(grouper)))
        citations_randomized.ix[cross_class_ind, ['Citing_Patent', 
                                                 'Cited_Patent']
                                ] = grouper.apply(randomize_citations_helper)
        
        ### Drop patent attributes (which are now inaccurate for both the citing and cited patent) and bring them in from patent_attributes
        citations_randomized = citations_randomized[['Citing_Patent', 'Cited_Patent', 'Same_Class']]

        citations_randomized = citations_randomized.merge(patent_attributes, 
                        left_on='Citing_Patent', 
                        right_index=True,
                        )

        citations_randomized = citations_randomized.merge(patent_attributes, 
                        left_on='Cited_Patent', 
                        right_index=True,
                        suffixes=('_Citing_Patent','_Cited_Patent'))
        return citations_randomized


#     @timeit
    def randomize_citations_helper(citing_cited):
        
        ind = citing_cited.index
        rewired_output = citing_cited.copy()


        Citing_lookup = pd.Series(index=citing_cited.Citing_Patent.unique(),
                                  data=1+arange(citing_cited.Citing_Patent.nunique()))
        Cited_lookup = pd.Series(index=citing_cited.Cited_Patent.unique(),
                                 data=1+arange(citing_cited.Cited_Patent.nunique()))

        n_Citing = len(Citing_lookup)
        n_Cited = len(Cited_lookup)
#         print(n_Citing*n_Cited)
        if n_Cited*n_Citing==len(ind): #The graph is fully connected, and so can't be rewired
            return rewired_output

        citing_cited.Citing_Patent = Citing_lookup.ix[citing_cited.Citing_Patent].values
        citing_cited.Cited_Patent = Cited_lookup.ix[citing_cited.Cited_Patent].values
        citing_cited.Cited_Patent += n_Citing

#         ro.globalenv['citing_cited'] = ro.Vector(citing_cited.values.ravel(order='C'))
#         ro.globalenv['n_Citing'] = ro.default_py2ri(n_Citing)
#         ro.globalenv['n_Cited'] = ro.default_py2ri(n_Cited)    
#         ro.r('g = graph.bipartite(c(rep(T, n_Citing), rep(F, n_Cited)), citing_cited)')
#         ro.r('h = birewire.rewire.bipartite(g, verbose=FALSE, exact=TRUE)')
#         z = array(ro.r('z = get.edgelist(h)')).astype('int')
        this_rewiring = br.Rewiring(data=citing_cited.values,
                                   type_of_array='edgelist_b',
                                   type_of_graph='bipartite')
        this_rewiring.rewire(verbose=0)   
        z = this_rewiring.data_rewired


        Citing_lookup = pd.DataFrame(Citing_lookup).reset_index().set_index(0)
        Cited_lookup = pd.DataFrame(Cited_lookup).reset_index().set_index(0)

        citing_patents = Citing_lookup.ix[z[:,0]].values.flatten()
        cited_patents = Cited_lookup.ix[z[:,1]-n_Citing].values.flatten()
        
#         df = pd.DataFrame(index=ind,
#                          columns=['Citing_Patent', 'Cited_Patent'],
#                          )
        rewired_output['Citing_Patent'] = citing_patents
        rewired_output['Cited_Patent'] = cited_patents
        return rewired_output#citing_patents, cited_patents


# In[12]:

# all(citations.Citing_Patent.value_counts()==citations_rewired.Citing_Patent.value_counts())
# all(citations.Cited_Patent.value_counts()==citations_rewired.Cited_Patent.value_counts())
# all(citations.Year_Citing_Patent.value_counts()==citations_rewired.Year_Citing_Patent.value_counts())
# all(citations.Year_Cited_Patent.value_counts()==citations_rewired.Year_Cited_Patent.value_counts())
# all(citations.Class_Cited_Patent.value_counts()==citations_rewired.Class_Cited_Patent.value_counts())
# all(citations.Class_Citing_Patent.value_counts()==citations_rewired.Class_Citing_Patent.value_counts())
# all(citations.groupby(['Year_Citing_Patent', 'Year_Cited_Patent'])['Same_Class'].count() == citations_rewired.groupby(['Year_Citing_Patent', 'Year_Cited_Patent'])['Same_Class'].count())


# In[13]:

# n_erroneous_cross_class = (citations_rewired['Class_Cited_Patent']==citations_rewired['Class_Citing_Patent'] * ~citations_rewired['Same_Class']).sum()
# n_erroneous_cross_class/citations.shape[0]


# In[14]:

### Establish metrics and how to calculate them

from sklearn.metrics import pairwise_distances

@timeit
def cosine_similarities(citation_counts):
    similarities = 1-pairwise_distances(citation_counts, metric="cosine")
    
#     #In case there are any classes not covered in this citation count matrix, they will 
#     #be the ones at the end (e.g. if there are 430 classes, this citation count matrix could
#     #only go up to 420 classes, in which case the remaining 10 classes should all have 0s)
#     all_similarities = zeros((max(classes)+1, max(classes)+1))
#     all_similarities[:present_similarities.shape[0], 
#                      :present_similarities.shape[1]] = present_similarities
    
    return pd.DataFrame(data=similarities,
                        columns=classes,
                        index=classes)


@timeit
def jaccard_similarities(citation_counts):
    return pd.DataFrame(data=pairwise_distances(citation_counts>0, metric=jaccard_helper),
                        columns=classes,
                        index=classes)

from scipy.sparse import find as sfind
def jaccard_helper(x,y):
    I, J, V = sfind(x)
    I1, J1, V1 = sfind(y)
    J = set(J)
    J1 = set(J1)
    try:
        return len(J.intersection(J1))/len(J.union(J1))
    except ZeroDivisionError:
        return 0


@timeit
def calculate_citation_counts(citations, 
                              relation='class_cites_class',
                              up_to_year=False):
    if up_to_year and up_to_year!='all':
        citations = citations[citations['Year_Citing_Patent']<=up_to_year]


    if relation=='class_cites_class':
        ### Calculate citation counts from each class to each class
        citation_counts = citations.groupby(['Class_Citing_Patent', 'Class_Cited_Patent'
                                          ])['Citing_Patent'].count()        
        
        citation_counts = pd.DataFrame(citation_counts)
        citation_counts.rename(columns={'Citing_Patent': 'Count'}, inplace=True)
        citation_counts.reset_index(inplace=True)
        val = citation_counts['Count'].values
        x = citation_counts['Class_Citing_Patent'].values
        y = citation_counts['Class_Cited_Patent'].values
        dims = (len(classes), len(classes))
        
    elif relation=='patent_cites_class':
        ### Calculate citation counts from each patent to each class
        citation_counts = citations.groupby(['Citing_Patent', 'Class_Cited_Patent', 
                                              ])['Citing_Patent'].count()
        
        citation_counts = pd.DataFrame(citation_counts)
        citation_counts.rename(columns={'Citing_Patent': 'Count'}, inplace=True)
        citation_counts.reset_index(inplace=True)
        val = citation_counts['Count'].values
        x = citation_counts['Citing_Patent'].values
        y = citation_counts['Class_Cited_Patent'].values
        dims = (max(x)+1, len(classes))
        
    elif relation=='class_cites_patent':
        ### Calculate citation counts from each class to each patent
        citation_counts = citations.groupby(['Cited_Patent', 'Class_Citing_Patent', 
                                          ])['Cited_Patent'].count()
        #Note: Typical convention is to read FROM rows TO columns (i.e. the arrows or citations go from the row value)
        #to the column value. This dataframe breaks that convention (i.e. does the reverse). The reason for this is
        #that in pandas it is easier to have a bazillion rows than to have a bazillion columns. Since we have far more
        #individual patents than individual classes (order of 4 million vs order of 100), we are making the patents the
        #rows and the classes the columns. So, be careful when using this output in the future. Be sure to transpose it
        #when needed!
        
        citation_counts = pd.DataFrame(citation_counts)
        citation_counts.rename(columns={'Cited_Patent': 'Count'}, inplace=True)
        citation_counts.reset_index(inplace=True)
        val = citation_counts['Count'].values
        x = citation_counts['Cited_Patent'].values
        y = citation_counts['Class_Citing_Patent'].values
        dims = (max(x)+1, len(classes))
    
    from scipy.sparse import csr_matrix
    citation_counts = csr_matrix((val, (x,y)), shape=dims)
    
#     citation_counts = citation_counts.unstack()
#     citation_counts.sort(axis=1,inplace=True)    
#     citation_counts.sort(axis=0,inplace=True)    
#     citation_counts.fillna(0, inplace=True)

    return citation_counts


# In[15]:

#classes = sort(list(set(citations['Class_Cited_Patent'].unique()).union(citations['Class_Citing_Patent'].unique())))
classes = arange(len(class_lookup))
years = set(citations['Year_Cited_Patent'].unique()).union(citations['Year_Citing_Patent'].unique())
years = list(range(min(years), max(years)+1))

if target_years is None or target_years=='all':
    target_years = years


# In[16]:

@timeit
def cocitation_counts(citations, 
                      up_to_year=False):
    if up_to_year and up_to_year!='all':
        citations = citations[citations['Year_Citing_Patent']<=up_to_year]

    import scipy.sparse
    patent_class_citations = scipy.sparse.csr_matrix((ones_like(citations['Citing_Patent']),
                                                      (citations['Citing_Patent'].values, 
                                                       citations['Class_Cited_Patent'].values)))

    present_cocitation_counts = (patent_class_citations.T * patent_class_citations).todense()
    
    all_cocitation_counts = zeros((max(classes)+1, max(classes)+1))
    all_cocitation_counts[:present_cocitation_counts.shape[0], 
                          :present_cocitation_counts.shape[1]] = present_cocitation_counts
    
    return all_cocitation_counts


# In[17]:

def calculate_citation_networks(citations,
                                metrics,
                                target_years,
                                classes=classes,
                                n_years=n_years
                                ):
    networks = pd.Panel4D(labels=metrics,
                          items=target_years,
                          major_axis=classes,
                          minor_axis=classes,
                          dtype='float64')

    for year in target_years:
        print(year)
#         these_citations = citations[citations['Year_Citing_Patent']<=year]
        if n_years is None or n_years=='all' or n_years=='cumulative':
            these_citations = citations[citations['Year_Citing_Patent']<=year]
        else:
            these_citations = citations[((citations['Year_Citing_Patent']<=year) & 
                                           (citations['Year_Citing_Patent']>(year-n_years)))]
        
        if 'Class_CoCitation_Count' in metrics:
            print('Class_CoCitation_Count')
            networks.ix['Class_CoCitation_Count', year,:,:] = cocitation_counts(these_citations)
        
        citation_counts = calculate_citation_counts(these_citations,
                                                    relation='class_cites_class')
        if 'Class_Cites_Class_Count' in metrics:
            print('Class_Cites_Class_Count')
            networks.ix['Class_Cites_Class_Count', year,:,:] = array(citation_counts.todense())
        if 'Class_Cited_by_Class_Count' in metrics:
            print('Class_Cited_by_Class_Count')
            networks.ix['Class_Cited_by_Class_Count', year,:,:] = array(citation_counts.todense().T)
        
        if 'Class_Cites_Class_Input_Cosine_Similarity' in metrics:
            print('Class_Cites_Class_Input_Cosine_Similarity')
            networks.ix['Class_Cites_Class_Input_Cosine_Similarity', year,:,:] = cosine_similarities(citation_counts)
        if 'Class_Cites_Class_Output_Cosine_Similarity' in metrics:
            print('Class_Cites_Class_Output_Cosine_Similarity')
            networks.ix['Class_Cites_Class_Output_Cosine_Similarity', year,:,:] = cosine_similarities(citation_counts.T)

        citation_counts = calculate_citation_counts(these_citations,
                                                    relation='class_cites_patent')
        if 'Class_Cites_Patent_Input_Cosine_Similarity' in metrics:
            print('Class_Cites_Patent_Input_Cosine_Similarity')
            networks.ix['Class_Cites_Patent_Input_Cosine_Similarity', year,:,:] = cosine_similarities(citation_counts.T)
        
        if 'Class_Cites_Patent_Input_Jaccard_Similarity' in metrics:
            print('Class_Cites_Patent_Input_Jaccard_Similarity')
            networks.ix['Class_Cites_Patent_Input_Jaccard_Similarity', year,:,:] = jaccard_similarities(citation_counts.T)

        citation_counts = calculate_citation_counts(these_citations,
                                                    relation='patent_cites_class')
        if 'Patent_Cites_Class_Output_Cosine_Similarity' in metrics:
            print('Patent_Cites_Class_Output_Cosine_Similarity')
            networks.ix['Patent_Cites_Class_Output_Cosine_Similarity', year,:,:] = cosine_similarities(citation_counts.T)
        
        if 'Patent_Cites_Class_Output_Jaccard_Similarity' in metrics:
            print('Patent_Cites_Class_Output_Jaccard_Similarity')
            networks.ix['Patent_Cites_Class_Output_Jaccard_Similarity', year,:,:] = jaccard_similarities(citation_counts.T)
        

    return networks


# In[18]:

if randomized_control:
    citations = randomize_citations(citations)
    print("Time until randomizations are done: %.2f"%(time()-t))


# In[19]:

# citations_rewired = randomize_citations(citations)
# counts_empirical = calculate_citation_counts(citations, relation='class_cites_class').todense()
# counts_rewired = calculate_citation_counts(citations_rewired, relation='class_cites_class').todense()
# networks_rewired = calculate_citation_networks(citations_rewired, metrics, target_years)


# In[20]:

networks = calculate_citation_networks(citations, citation_metrics, target_years)


# In[21]:

print("Time until calculations are done: %.2f"%(time()-t))


# In[22]:

networks.major_axis = class_lookup.index[networks.major_axis]
networks.minor_axis = class_lookup.index[networks.minor_axis]


# In[23]:

if randomized_control:
    file_name = 'synthetic_control_citations_%s%s_%i.h5'%(years_label, class_system, randomization_id)
    
#     file_name = '5_years_'+file_name
    
    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/citations/controls/%s/%s'%(class_system,file_name),
                    mode='w', table=True)
    store.put('/synthetic_citations_'+class_system, networks, 'table', append=False)
    store.close()
else:
    store = pd.HDFStore(data_directory+'Class_Relatedness_Networks/citations/class_relatedness_networks_citations.h5',
                        mode='a', table=True)
    store.put('/empirical_citations_'+years_label+class_system, networks, 'table', append=False)
    store.close()


# In[24]:

print("Total runtime: %.2f"%(time()-t))

