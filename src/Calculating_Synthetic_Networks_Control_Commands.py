
# coding: utf-8

# In[ ]:

# python_location = '/home/jeffrey_alstott/anaconda3/bin/python'


# In[6]:

# class_system = 'IPC4'


# In[8]:

# basic_program = open('Calculating_Citation_Networks.py', 'r').read()
# basic_program = open('Calculating_CoOccurrence_Networks.py', 'r').read()


# In[9]:

# data_directory = '../data/'
# job_type = 'citations'
# job_type = 'cooccurrence'
# entity = 'Firm'
job_label = job_type

try:
    job_label += '_'+entity
except NameError:
    pass


# In[16]:

# first_rand_id = None
# n_randomizations = 1000

if overwrite:
    runs = range(first_rand_id,first_rand_id+n_randomizations)
else:
    from os import listdir
    dirlist = listdir(data_directory+'Class_Relatedness_Networks/%s/controls/%s/'%(job_type,class_system))
    from pylab import *
    unrun_iterations = ones(n_randomizations)

    for f in dirlist:
        if f.startswith('synthetic_control'):
            n = int(f.split('_')[-1][:-3])
            unrun_iterations[n] = 0

    unrun_iterations = where(unrun_iterations)[0]

    runs = unrun_iterations


# In[12]:

from os import system

for randomization_id in runs:
    header = """#!{3}
#PBS -l nodes=1:ppn=1
#PBS -l walltime=3:00:00
#PBS -l mem=20000m
#PBS -N rand_{0}_{1}_{2}

randomization_id = {0}
print("Randomization number: %i"%randomization_id)
""".format(randomization_id, class_system, job_label, python_location)

#     options = """
#     class_system = '{1}'
# target_years = 'all'
# n_years = 'all'

# data_directory = '{3}'
# randomized_control = True
# n_years = None
# # preserve_years = True
# # chain = False

# # occurrence_data = 'occurrences_organized.h5'
# # entity_data = 'entity_classes_Firm'
# # entity_column = 'Firm'
# # occurrence_data = 'classifications_organized.h5'
# # entity_data = 'patent_classes'
# # entity_column = 'PID'
# # print(occurrence_data)
# # print(entity_data)
# # print(entity_column)
# """
    this_program = header+options+basic_program

    this_job_file = 'jobfiles/randomization_{0}_{1}_{2}.py'.format(randomization_id, class_system, job_label)


    f = open(this_job_file, 'w')
    f.write(this_program)
    f.close()

    system('qsub '+this_job_file)

