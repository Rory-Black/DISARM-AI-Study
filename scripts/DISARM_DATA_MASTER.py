# DISARM class from https://github.com/DISARMFoundation/DISARMframeworks/blob/main/CODE/generate_DISARM_pages.py#L250
import pandas as pd
import numpy as np
import os
class DISARMDataMaster:
    
    def __init__(self, 
                 frameworkfile = 'DISARM_FRAMEWORKS_MASTER.xlsx', 
                 datafile = 'DISARM_DATA_MASTER.xlsx',
                 commentsfile = 'DISARM_COMMENTS_MASTER.xlsx'):
        
        # Load metadata from file
        metadata = {}
        xlsx = pd.ExcelFile(frameworkfile)
        for sheetname in xlsx.sheet_names:
            metadata[sheetname] = xlsx.parse(sheetname)
            metadata[sheetname].replace(np.nan, '', inplace=True)

        xlsx = pd.ExcelFile(datafile)
        for sheetname in xlsx.sheet_names:
            metadata[sheetname] = xlsx.parse(sheetname)
            metadata[sheetname].replace(np.nan, '', inplace=True)

        # Create individual tables and dictionaries
        self.df_phases = metadata['phases']
        self.df_frameworks = metadata['frameworks']
        self.df_techniques = metadata['techniques']
        self.df_tasks = metadata['tasks']
        self.df_incidents = metadata['incidents']
        self.df_urls = metadata['urls']
        #self.df_urls['url_id'] = self.df_urls['url_id'].str.rstrip # strip trailing spaces from urls to allow merge to work
        self.df_externalgroups = metadata['externalgroups']
        self.df_tools = metadata['tools']
        self.df_examples = metadata['examples']
        self.df_counters = metadata['countermeasures'].sort_values('disarm_id')
        self.df_counters[['tactic_id', 'tactic_name']] = self.df_counters['tactic'].str.split(' ', n=1, expand=True)
        self.df_counters[['metatechnique_id', 'metatechnique_name']] = self.df_counters['metatechnique'].str.split(' ', n=1, expand=True)
        self.df_detections = metadata['detections']
        self.df_detections[['tactic_id', 'tactic_name']] = self.df_detections['tactic'].str.split(' ', n=1, expand=True)
#        self.df_detections[['metatechnique_id', 'metatechnique_name']] = self.df_detections['metatechnique'].str.split(' ', n=1, expand=True) #FIXIT
        self.df_actortypes = metadata['actortypes']
        self.df_resources = metadata['resources']
        self.df_responsetypes = metadata['responsetypes']
        self.df_metatechniques = metadata['metatechniques']
        self.it = self.create_incident_technique_crosstable(metadata['incidenttechniques'])
        self.at = self.create_associated_techniques_crosstable(metadata['associatedtechniques'])
        self.df_tactics = metadata['tactics']
        self.df_playbooks = metadata['playbooks']
        self.df_sectors = metadata['sectors']

        # Add columns containing lists of techniques and counters to the tactics dataframe
        self.df_techniques_per_tactic = self.df_techniques.groupby('tactic_id')['disarm_id'].apply(list).reset_index().rename({'disarm_id':'technique_ids'}, axis=1)
        self.df_counters_per_tactic = self.df_counters.groupby('tactic_id')['disarm_id'].apply(list).reset_index().rename({'disarm_id':'counter_ids'}, axis=1)
        self.df_tactics = self.df_tactics.merge(self.df_techniques_per_tactic, left_on='disarm_id', right_on='tactic_id', how='left').fillna('').drop('tactic_id', axis=1)
        self.df_tactics = self.df_tactics.merge(self.df_counters_per_tactic, left_on='disarm_id', right_on='tactic_id', how='left').fillna('').drop('tactic_id', axis=1)

        # Add simple dictionaries (id -> name) for objects
        self.phases      = self.make_object_dictionary(self.df_phases)
        self.tactics     = self.make_object_dictionary(self.df_tactics)
        self.techniques  = self.make_object_dictionary(self.df_techniques)
        self.counters    = self.make_object_dictionary(self.df_counters)
        self.metatechniques = self.make_object_dictionary(self.df_metatechniques)
        self.responsetypes = self.make_object_dictionary(self.df_responsetypes)
        self.actortypes  = self.make_object_dictionary(self.df_actortypes)
        self.resources   = self.make_object_dictionary(self.df_resources)
        self.sectors     = self.make_object_dictionary(self.df_sectors)

        # Create the data table for each framework file
        self.num_tactics = len(self.df_tactics)

        # Create counters and detections cross-tables
        self.cross_counterid_techniqueid = self.create_cross_table(self.df_counters[['disarm_id', 'techniques']], 
                                                                   'techniques', 'technique', '\n')        
        self.cross_counterid_resourceid = self.create_cross_table(self.df_counters[['disarm_id', 'resources_needed']], 
                                                                  'resources_needed', 'resource', ',')
        self.cross_counterid_actortypeid = self.create_cross_table(self.df_counters[['disarm_id', 'actortypes']], 
                                                                  'actortypes', 'actortype', ',')
        self.cross_detectionid_techniqueid = self.create_cross_table(self.df_detections[['disarm_id', 'techniques']], 
                                                                   'techniques', 'technique', '\n')        
        self.cross_detectionid_resourceid = self.create_cross_table(self.df_detections[['disarm_id', 'resources_needed']], 
                                                                  'resources_needed', 'resource', ',')
        self.cross_detectionid_actortypeid = self.create_cross_table(self.df_detections[['disarm_id', 'actortypes']], 
                                                                  'actortypes', 'actortype', ',')
        self.cross_incidentid_urls = self.create_cross_table(self.df_incidents[['disarm_id', 'urls']], 
                                                                  'urls', 'url', ' ')      

    def create_incident_technique_crosstable(self, it_metadata):
        # Generate full cross-table between incidents and techniques

        it = it_metadata
        it.index=it['disarm_id']
        it = it['technique_ids'].str.split(',').apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('technique_id').reset_index().merge(it.drop('disarm_id', axis=1).reset_index()).drop('technique_ids', axis=1)
        it = it.merge(self.df_incidents[['disarm_id','name']], 
                      left_on='incident_id', right_on='disarm_id',
                      suffixes=['','_incident']).drop('incident_id', axis=1)
        it = it.merge(self.df_techniques[['disarm_id','name']], 
                      left_on='technique_id', right_on='disarm_id',
                      suffixes=['','_technique']).drop('technique_id', axis=1)
        return(it)

    def create_associated_techniques_crosstable(self, at_metadata):
        # Generate full cross-table between associated techniques and techniques

        at = at_metadata
        at.index=at['disarm_id']
        at = at.merge(self.df_techniques[['disarm_id','name']],
                    left_on='associated_technique_id', right_on='disarm_id',
                    suffixes=['','_associated']).drop('associated_technique_id', axis=1)
        at = at.merge(self.df_techniques[['disarm_id', 'name']],
                    left_on='technique_id', right_on='disarm_id',
                    suffixes=['','_technique']).drop('technique_id', axis=1)
        return(at)

    def make_object_dictionary(self, df):
        return(pd.Series(df.name.values,index=df.disarm_id).to_dict())


    def create_cross_table(self, df, col, newcol, divider=','):
        ''' Convert a column with multiple values per cell into a crosstable

        # Thanks https://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows?noredirect=1
        '''
        crosstable = df.join(df[col]
                        .str.split(divider, expand=True).stack()
                        .reset_index(drop=True,level=1)
                        .rename(newcol)).drop(col, axis=1)
        crosstable = crosstable[crosstable[newcol].notnull()]
        crosstable[newcol+'_id'] = crosstable[newcol].str.split(' ').str[0]
        crosstable.drop(newcol, axis=1, inplace=True)
        return crosstable

    # OWN WORK =============================================================

    def get_incident_ids(self):
        ids = self.df_incidents['disarm_id']
        return ids.tolist()
    
    def get_incident_urls(self, incidentid):
        urls = []
        incidentid_urls = self.cross_incidentid_urls[self.cross_incidentid_urls['disarm_id']==incidentid]
        incidentid_urls = pd.merge(incidentid_urls, self.df_urls[['url_id', 'pub_date', 'authors', 'org', 'archive_link']])
        for index, row in incidentid_urls.iterrows():
            urls.append(row['archive_link'])
        return urls    
    
    def get_incident_techniques(self, incidentid):
        techniques = []
        techlist = self.it[self.it['disarm_id_incident'] == incidentid]
        for index, row in techlist.sort_values('disarm_id_technique').iterrows():
            techniques.append(row['disarm_id_technique'])
        return techniques
    
    def get_incident_techniques_with_desc(self, incidentid):
        techniques = []

        techlist = self.it[self.it['disarm_id_incident'] == incidentid]
        for index, row in techlist.sort_values('disarm_id_technique').iterrows():
            techniques.append((
                row['disarm_id_technique'],
                row['name']
            ))

        return techniques
    