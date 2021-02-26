import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
from sklearn.base import BaseEstimator, TransformerMixin

class data_null_transform(BaseEstimator, TransformerMixin):
    def __init__(self,colum='GEBURTSJAHR'):
        self.data_dic_dir = 'data_transform/cols_actions.xlsx'
        self.dic_geburtsjahr = None

    def read_actions_csv(self,df_columns):
        '''
            read information from cols_actions.xlsx and add the columns created 
            in previous steps

            Parameters:
                df_columns(Array): Names of the columns to find in cols_actions.xlsx

            Return
                DataFrame informacion for the parameter cols in the cols_actions.xlsx
        '''
        df_col_actions = pd.read_excel(self.data_dic_dir)

        data = {'COLUMN':['PRAEGENDE_JUGENDJAHRE_DECADE','PRAEGENDE_JUGENDJAHRE_MOVEMENT'],
        'CATEGORY':['ordinal','categorical'],
        'ACTION':['keep','keep'],
        'OBS':[np.nan,np.nan]
        }

        df_col_actions = df_col_actions.append(pd.DataFrame(data), ignore_index= True)
        df_col_actions.tail()


        data = {'COLUMN':['PLZ8_BAUMAX_FAMILY_BUSINESS'],
                'CATEGORY':['categorical'],
                'ACTION':['keep'],
                'OBS':[np.nan]
        }

        df_col_actions = df_col_actions.append(pd.DataFrame(data), ignore_index= True)

        data = {'COLUMN':['WOHNLAGE_CITY_RURAL'],
                'CATEGORY':['categorical'],
                'ACTION':['keep'],
                'OBS':[np.nan]
        }

        df_col_actions = df_col_actions.append(pd.DataFrame(data), ignore_index= True)
        
        df_col_actions = df_col_actions[df_col_actions['COLUMN'].isin(df_columns)]
        
        return df_col_actions
        
    def fillna_lotto(self,X):
        '''
            column to deal with missing values for D19_LOTTO column
            
            Parameters:
                    X(Dataframe): whole data DataFrame to transform
                    
            Returns:
                DataFrame with information transformed
        '''
        X['D19_LOTTO'].fillna(0, inplace= True)
        return X
    
    def fit_geburstjahr(self,X):
        '''
            generate the transformation dictionary for geburstjahr column
            which is the mean geburstjahr for each PRAEGENDE_JUGENDJAHRE_DECADE
            value
            
            Parameters:
                X(Dataframe): whole data DataFrame to transform
                
            Return:
                dic: dictionary with average age per PRAEGENDE_JUGENDJAHRE_DECADE
        '''
        c0 = X['GEBURTSJAHR'] > 0
        dic = X[c0].groupby('PRAEGENDE_JUGENDJAHRE_DECADE')[['GEBURTSJAHR']].mean().round().to_dict()
        dic = dic['GEBURTSJAHR']
        self.dic_geburtsjahr = dic
        
        return dic
    
    def fillna_geburstjahr(self,X):
        '''
            fill empty values with two strategies:
            1. average year of the decade stored in PRAEGENDE_JUGENDJAHRE_DECADE
            2. if it does not have decade fill with the average year in total
            
            Parameters:
                X(Dataframe): whole data DataFrame to transform
                
            Returns:
                DataFrame with information transformed
        '''
        
        c0 = X['GEBURTSJAHR'] > 0
        # First strategie
        X.loc[~c0,'GEBURTSJAHR'] = X.loc[~c0,'PRAEGENDE_JUGENDJAHRE_DECADE'].replace(self.dic_geburtsjahr)

        #apply mean to missing values 
        X.fillna(X['GEBURTSJAHR'].mean().round(),inplace=True)
        
        return X 
        
    def fillna_ordinal_categorical(self,X):
        '''
            fill null values of categorical and ordinal data
            
            Parameters:
                X(Dataframe): whole data DataFrame to transform
                
            Returns:
                DataFrame with information transformed
        '''
        
        new_missing_category_cols = self.df_col_actions.loc[self.df_col_actions['CATEGORY'].isin(['ordinal','categorical']),'COLUMN']
        X[new_missing_category_cols] = X[new_missing_category_cols].fillna(-1)
        
        return X
        
    def fit(self, X, y=None):
        
        self.dic_geburtsjahr = self.fit_geburstjahr(X)
        X_cols = X.columns.values
        self.df_col_actions = self.read_actions_csv(X_cols)
        
        return self

    def transform(self, X):
        print(X.shape)
        X_ = X.copy()
        print(X_.shape)
        X_= self.fillna_lotto(X_)
        X_= self.fillna_geburstjahr(X_)
        X_= self.fillna_ordinal_categorical(X_)
        
        return X_

def create_nan_dict(dir_attribute_file):
    '''
        create a dictionary in the form
        {
            column_name: {key of nan value: np.nan}
        }
        
        by taking as reference for the keys the file stated as attribute
        
        parameters:
            dir_attribute_file (str): location of column descriptive file
        
        returns
            (dict): dictionary as stated before
    '''    
    
    
    df_atributes = pd.read_excel(dir_attribute_file)

    # clean df
    df_atributes.drop(columns='Unnamed: 0', inplace=True)
#     df_atributes.columns = df_atributes.iloc[0,:]
#     df_atributes = df_atributes.drop([0]).reset_index()
    df_atributes = df_atributes.fillna(method='ffill')

    # ways of identifying null encoded values
    c1 = df_atributes['Meaning'].str.contains('unknown')
    c2 = df_atributes['Meaning'].str.contains('no transaction|s known', regex=True)
    c3 = df_atributes['Meaning'].str.contains('no Online-transactions within the last 12 months')
    
    # create df of enconded nulls
    df_atributes_nan = df_atributes.loc[c1|c2|c3]

    '''
    create a dictionary in the form
        {
            column_name: {key of nan value: np.nan}
        }
    '''
    dict_col_key_nan = {}

    for c in df_atributes_nan['Attribute'].unique():
        c0 = df_atributes_nan['Attribute'] == c
        
        if '_RZ' in c:
            c = c.replace('RZ','')
        
        # get all the keys for each column an turn it into a dictionary
        key_val = df_atributes_nan.loc[c0,'Value'].\
                                        astype(str).str.replace(' ','').str.split(',').\
                                        values[0]

        dict_col_key_nan[c] = dict.fromkeys([int(x) for x in key_val], np.nan)
        
        if 'KBA05' in c:
            #change to toake into account similar columns on Udacity_AZDIAS_052018.csv file
            dict_col_key_nan[c.replace('05','13')] = dict.fromkeys([int(x) for x in key_val], np.nan)
            

    return dict_col_key_nan


def transform_key_to_nan(df):
    '''
        transform the keys that represent missing values to np.nan
        
        Parameters:
            df(DataFrame): dataframe to be transformed

        Returns
            DataFrame with all the keys transformed to np.nan
    '''
    
    #create column keys values
    dir_attribute_file = 'DIAS Attributes - Values 2017.xlsx'
    dict_col_key_nan = create_nan_dict(dir_attribute_file)

    # Replace values
    for k,v in dict_col_key_nan.items():
        try:
            print("success: replacing {} column keys to Nan".format(k))
            df[k] = df[k].replace(v)
        except KeyError:
            print("warning: following column not found: {}".format(k))
    
    return df 
    
def clean_columns_high_null(df):
    '''
        drop the columns with a high percentage of missing values, 
        plot the percentage of missing values of those columns
        
        Parameters:
            df(DataFrame): dataframe to be cleaned

        Returns
            DataFrame with high null columns droped 
    '''
    null_count = df.isna().mean().sort_values(ascending=False).reset_index()
    
    # set limit
    limit = 0.3

    c0 = null_count[0] >= limit

    # ploting results
    fig,ax = plt.subplots(figsize=(10,10))
    sns.barplot(x=0, y='index', data=null_count[c0], color='grey')
    fig.tight_layout()
    
    fig.savefig("Images/percentages_columns_missing_values.png")
    
    print("Dropping columns with high percetnage of missing values")
    
    drop_cols = null_count[c0]['index'].values
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    print("ammount of dropped columns: {}".format(len(drop_cols)))
    print("remaining columns: {}".format(len(df.columns.values)))

    print("END removing columns high percetnage of missing values")
    print()
    
    return df
    
def clean_columns_not_in_data_dic(df, data_dic_dir):
    '''
        perform the action stated in the xlsx file to clean data that where not avaliable in 
        the data dictionaries given for the project
        
        - DIAS Attributes - Values 2017.xlsx
        - DIAS Information Levels - Attributes 2017.xlsx
        
        Parameters:
            df(DataFrame): dataframe to be cleaned
            data_dic_dir(str): location of the cols_actions.xlsx file            

        Returns
            DataFrame with columns droped 


    '''
    
    df_col_actions = pd.read_excel(data_dic_dir)
    c0 = df_col_actions['ACTION'] == 'drop'

    drop_cols = df_col_actions.loc[c0,'COLUMN'].values

    print("removing columns not avaliable in the data dictionary")
    df.drop(columns=drop_cols, inplace= True, errors='ignore')
       
    print("ammount of dropped columns: {}".format(len(drop_cols)))
    print("remaining columns: {}".format(len(df.columns.values)))
    print()

    return df

def clean_redundant_columns(df):
    '''    
        this columns have others that are more general an therefore those will be used instead
        generaly the columns have GOB in their name

        Parameters:
            df(DataFrame): dataframe to be cleaned
        
        Returns
            DataFrame with columns droped 
    '''
    drop_cols = ['CAMEO_DEU_2015',
                 'LP_FAMILIE_FEIN',
                 'LP_LEBENSPHASE_FEIN',
                 'LP_STATUS_FEIN'
                ]
    
    print("removing redundat cols: {}".format(drop_cols))
    df.drop(columns=drop_cols, inplace= True, errors='ignore')
    print("ammount of dropped columns: {}".format(len(drop_cols)))
    print("remaining columns: {}".format(len(df.columns.values)))
    print()
    
    return df

def clean_id_column(df,id_column):
    '''
        remove columns which are unique un the dataframe 

        Parameters:
            df(DataFrame): dataframe to be cleaned
            id_column(Array): Arrray of id columns to be dropped
        
        Returns
            DataFrame with columns droped 
    '''
    print("removing id cols")
    print()
    
    return df.drop(columns=id_column)

def transform_OST_WEST_KZ(df):
    '''
        encode to numeric values 
        O : 0
        W : 1

        Parameters:
            df(DataFrame): dataframe to be cleaned
        
        Returns
            DataFrame with columns droped 
    '''
    
    print("START encoding OST_WEST_KZ column")
    df['OST_WEST_KZ'].replace(['O','W'], [0, 1], inplace=True)
    print("END encoding OST_WEST_KZ column")
    print()
    
    return df

def transform_PRAEGENDE_JUGENDJAHRE(df):
    '''
        transform PRAEGENDE_JUGENDJAHRE column into two different columns containging
        information about decade and generational movement
        
        PRAEGENDE_JUGENDJAHRE_DECADE
            1: 40 ies
            2: 50 ies
            3: 60 ies
            4: 70 ies
            5: 80 ies
            6: 90 ies
        
        PRAEGENDE_JUGENDJAHRE_MOVEMENT
            1: Mainstream
            0: Avantgrade 

        Parameters:
            df(DataFrame): dataframe to be cleaned
        
        Returns
            DataFrame with columns droped 
    '''
    print("START transforming PRAEGENDE_JUGENDJAHRE")
    
    #Generate Decade Column
    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = df['PRAEGENDE_JUGENDJAHRE']
    #dict encoding
    dic_praegende_jugend_decade = {1:1,
                                   2:1,
                                   3:2,
                                   4:2,
                                   5:3,
                                   6:3,
                                   7:3,
                                   8:4,
                                   9:4,
                                   10:5,
                                   11:5,
                                   12:5,
                                   13:5,
                                   14:6,
                                   15:6                           
    }

    df['PRAEGENDE_JUGENDJAHRE_DECADE'].replace(dic_praegende_jugend_decade, inplace=True)

    # Generate Movmenet column
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE']
    #dict encoding
    dic_praegende_jugend_m = {1:0,
                              2:1,
                              3:0,
                              4:1,
                              5:0,
                              6:1,
                              7:1,
                              8:0,
                              9:1,
                              10:0,
                              11:1,
                              12:0,
                              13:1,
                              14:0,
                              15:1                           
    }
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].replace(dic_praegende_jugend_m, inplace=True)

    
    # drop PRAEGENDE_JUGENDJAHRE column since is not necesary
    df.drop(columns=['PRAEGENDE_JUGENDJAHRE'],inplace= True, errors='ignore')
    print("END transforming PRAEGENDE_JUGENDJAHRE")
    print()

    return df

def transform_PLZ8_BAUMAX(df):
    '''
        generate a new column from PLZ8_BAUMAX_FAMILY_BUSINESS
        that indicates whether the building is fpr family or business
        
            0 : family
            1 : business
    '''
    print("START transforming PLZ8_BAUMAX")
    
    df['PLZ8_BAUMAX_FAMILY_BUSINESS'] = df['PLZ8_BAUMAX']
    dic_PLZ8_BAUMAX = {1:0,
                       2:0,
                       3:0,
                       4:0,
                       5:1
                      }
    
    df['PLZ8_BAUMAX_FAMILY_BUSINESS'] = df['PLZ8_BAUMAX_FAMILY_BUSINESS'].replace(dic_PLZ8_BAUMAX)
    print("END transforming PLZ8_BAUMAX")
    print()
    
    return df
    
def transform_WOHNLAGE(df):
    '''
        generate a new column from WOHNLAGE column, thet indicates
        whether the biulding is in the city, rural or no score
        
            0 : no score
            1 : city
            2 : rural

        Parameters:
            df(DataFrame): dataframe to be cleaned
        
        Returns
            DataFrame with columns droped 
    '''
    print("START transforming WOHNLAGE")
    
    df['WOHNLAGE_CITY_RURAL'] = df['WOHNLAGE']
    dic_PLZ8_BAUMAX = {1:1,
                       2:1,
                       3:1,
                       4:1,
                       5:1,
                       7:2,
                       8:2
                      }
    
    df['WOHNLAGE_CITY_RURAL'] = df['WOHNLAGE_CITY_RURAL'].replace(dic_PLZ8_BAUMAX)
    print("END transforming WOHNLAGE")
    print()
    
    return df
    
def transform_CAMEO_DEUG_2015(df):
    '''
        standarize the fields from CAMEO_DEUG_2015 column since in the reading it gives
        different datatypes, values will be transformed to float an non numeric values to np.nan

        Parameters:
            df(DataFrame): dataframe to be cleaned
        
        Returns
            DataFrame with columns droped 
    '''
    
    print("START transforming CAMEO_DEUG_2015")
    
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].replace('X',np.nan).astype(np.float64)
    
    print("END transforming CAMEO_DEUG_2015")
    print()
    
    return df

def clean_null_rows(df, null_limit):
    '''
        drop rows that have more than (null_limit) columns with null value

        Parameter:
            df(DataFrame): dataframe to be cleaned
            null_limit(int): number to limit the ammount of null columns 
        
        Returns
            DataFrame with columns droped 
    '''
    print("Start clenaing null rows")
    
    condition_null_cols = df.isna().sum(axis=1)

    # get null rows graph
    fig,ax = plt.subplots(figsize=(10,10))
    sns.distplot(condition_null_cols,kde=False)
    fig.tight_layout()
    fig.savefig("Images/ammount_null_on_rows.png")

    df = df.drop(index=df.index[condition_null_cols>null_limit])
    
    print("END clenaing null rows")
    
    return df


def clean_data(df):
    '''
        call to all clean procedures for the data

        Parameter:
            df(DataFrame): dataframe to be cleaned            
        
        Returns
            DataFrame: cleaned DataFrame
    '''
    print("Start cleaning data")
    
    df = transform_key_to_nan(df)

    df = clean_columns_high_null(df)

    df = clean_id_column(df,['LNR'])
    
    data_dic_dir = 'data_transform/cols_actions.xlsx'
    df = clean_columns_not_in_data_dic(df,data_dic_dir)

    df = clean_redundant_columns(df)

    df = clean_null_rows(df,50)    
    
    print("End cleaning data")
    print()
    
    return df
    
def transform_data(df):
    '''
        call to all transform procedures for the data

        Parameter:
            df(DataFrame): dataframe to be transformed            
        
        Returns
            DataFrame: transformed DataFrame
    '''
    print("Start transforming data")
    df = transform_CAMEO_DEUG_2015(df)
    df = transform_OST_WEST_KZ(df)
    df = transform_PLZ8_BAUMAX(df)
    df = transform_PRAEGENDE_JUGENDJAHRE(df)
    df = transform_WOHNLAGE(df)
    
    print("End transforming data")
    print()
    
    return df

def clean_transform_data(df):
    '''
        call to the clean and transform procedures    

        Parameter:
            df(DataFrame): dataframe to be processed
        
        Returns
            DataFrame: processed DataFrame
    '''
    df = clean_data(df)
    df = transform_data(df)
    
    return df