
import os
import re
import json
import sys
import ast
import pickle
import pathlib
from pathlib import Path
from string import digits
import traceback

import requests
import pandas as pd
import numpy as numpy
import urllib.parse
from unicodedata import normalize
from pyairtable import Api, Base, Table
from unidecode import unidecode


BASE_NAME = os.getenv("BASE_NAME")
API_KEY = os.getenv("API_KEY")

def clean_batch(batch):
    existing = set()
    cleaned_batch = []
    for record in batch:
        record_id = record.get('id')
        if record_id not in existing:
            cleaned_batch.append(record)
            existing.add(record_id)
    return cleaned_batch


def create_or_update_airtable(create,start_with,data,fields_to_update,table_name,photo_fields=None): 
        
    if photo_fields is None: 
        photo_fields = []
    BASE_URL = 'https://api.airtable.com/v0/'
    api_key = API_KEY
    base_name = BASE_NAME
    api = Api(api_key)
    
    # Loop through the data DataFrame starting from the index 'start_with'
    
    for i in range(start_with,len(data)):
      
        records = data.iloc[i]
      
    
        # Loop through the fields that need to be updated or created
        for field in fields_to_update:
            if create==False:
                if field in photo_fields:
                    api.update(typecast=True,
                               record_id=records['record_id'], fields={field: ast.literal_eval(add_url_upload_format(records))})
                else:
                    try:
                        api.update(typecast=True,
                                   record_id=records['record_id'], fields={field: records.fillna('')[field]})
                    except TypeError:
                        api.update(typecast=True, record_id=records['record_id'], fields={field: str(records.fillna('')[field])})  
                        
            if create==True:   
                if field in photo_fields:
                    api.create(typecast=True,
                                fields={field: ast.literal_eval(add_url_upload_format(records))})
                else:
                    try:
                        api.create(typecast=True,
                                  fields={field: records.fillna('')[field]})
                    except TypeError:
                        api.create( typecast=True, fields={field: str(records.fillna('')[field])})
                        

def create_airtable(data,fields_to_update,table_name,start_with=0): 
        
    # if error, ask to stop or start from new
    try:
        create_or_update_airtable(create=True, start_with=start_with, data=data, fields_to_update=fields_to_update,table_name=table_name)
        return 'Success!'
    except Exception as e:
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])
        user_input = input( """ It through an error for : {}
        Where do you want to start again? If you want to stop, write n""".format( e))
        
        # If user does not want to stop, retry the creation process from the specified index
        if user_input != 'n':
            start_with = user_input
            create_or_update_airtable(create=True, start_with=int(user_input), data=data, fields_to_update=fields_to_update, base_name=base_name, table_name=table_name)
        else:
            return 'Error please review'
            
def update_airtable(data,fields_to_update,table_name,start_with=0): 
    # if error, ask to stop or start from new
    try:
        create_or_update_airtable(create=False, start_with=start_with, data=data, fields_to_update=fields_to_update,table_name=table_name)
        return 'Success!'
    except:
        return 'Error uploading to Airtable please review'
            

def format_dictionary_toAirtableFormat(dict1,is_create=False):
    # format dict of records to be updated into Airtable format from Dataframe format dict generated
    if is_create == False:
        dict2 = []    
        new_item = {
            "id": dict1["record_id"],
            "fields": {}
        }
        for key, value in dict1.items():
            if key != "record_id":
                new_item["fields"][key] = value
        dict2.append(new_item)
        return dict2[0]
    if is_create == True:
        dict2 = []    
        new_item = {}
        for key, value in dict1.items():
            if key != "record_id":
                new_item[key] = value
        dict2.append(new_item)
        return dict2[0]
        
        
def batch_create_or_update_airtable(create,data,fields_to_update,table_name,start_with=0):    
    BASE_URL = 'https://api.airtable.com/v0/'
    api_key = API_KEY
    base_name = BASE_NAME
    api = Api(api_key)
    table = api.table(base_name, table_name)
    # If updating, 'record_id' is essential
    if create==False:
        fields_to_update += ['record_id']
        
    # Create formatted batches of up to 10 records each from the input data using format_dictionary_toAirtableFormat
    formatted_batches=[]
    for i in range(start_with*10, len(data), 10):
        batch = data[fields_to_update].iloc[i:i+10]
        batch_as_dicts = batch.to_dict(orient='records')
        formatted_batch = [format_dictionary_toAirtableFormat(row,is_create=create) for row in batch_as_dicts]
        formatted_batches.append(formatted_batch)
    
    # Process each batch using Airtable API

    for batch in formatted_batches:
        if create==False:
            table.batch_update(typecast=True,records=clean_batch(batch))
    
        if create==True:
            table.batch_create(typecast=True,
                                               records=clean_batch(batch))
        start_with += 1


def batch_create_airtable(data,fields_to_update,table_name,start_with=0): 
    # if error, ask to stop or start from new
    try:
        batch_create_or_update_airtable(create=True, start_with=start_with, data=data, fields_to_update=fields_to_update,table_name=table_name)
        return 'Success!'
    except Exception as e:
        
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])
        
        user_input = input( """ It through an error for : {}
        Where do you want to start again? If you want to stop, write n""".format( e))
        
        if user_input != 'n':
            start_with = user_input
            create_or_update_airtable(create=True, start_with=int(user_input), data=data, fields_to_update=fields_to_update, base_name=base_name, table_name=table_name)
        else:
            return 'Error please review'
            
def batch_update_airtable(data,fields_to_update,table_name,start_with=0): 
    # if error, ask to stop or start from new
    try:
        batch_create_or_update_airtable(create=False, start_with=start_with, data=data, fields_to_update=fields_to_update,table_name=table_name)
        return 'Success!'
    except Exception as e:
        
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])
        
        user_input = input( """It through an error for : {}
        Where do you want to start again? If you want to stop, write n""".format( e))
        
        if user_input != 'n':
            start_with = user_input
            batch_create_or_update_airtable(create=False, start_with=int(user_input), data=data, fields_to_update=fields_to_update, table_name=table_name)
        else:
            return 'Error please review'

def download_airtable(table_name,view_name=False):
    BASE_URL = 'https://api.airtable.com/v0/'
    api_key = API_KEY
    base_name = BASE_NAME

    api = Api(api_key)
    #api.all(base_name, table_name)
    base = Base(api_key, base_name)
    #base.all('table_name')
    table = api.table(base_name, table_name)
    #table.all()
    if view_name:  
        df = table.all(view=view_name)
    else:
        df = table.all()
    df = pd.DataFrame(df)
    return df

def airtable_df_to_python(table_name,view_name=False):
    df = download_airtable(table_name=table_name,view_name=view_name)
    lista_records = list(df['id'])
    df = pd.DataFrame(list(df['fields']))
    df['record_id'] = lista_records
    return df

#limpiamos strings de fincas y de las calles de Madrid para tener el mismo formato
  
def columns_to_include_and_conversion_type(df):
    fields_to_update = []
    field_type = []
    fields_to_test = df.columns.to_list()
    for field in fields_to_test:
        if field == 'record_id':
            continue
        records = df.reset_index().iloc[0]
        try:
            api.create( fields={field: records[field]})
            fields_to_update.append(field)
            field_type.append('ok')
            
        except Exception as e:
            if 'the field is computed' in str(e) or 'not JSON' in str(e):
                continue

            if 'the field is computed' not in str(e) and 'cannot accept the provided value' in str(e) :
                try:
                    api.create( fields={field: str(records[field])})
                    fields_to_update.append(field)
                    field_type.append('str')
                except:
                    try:
                        api.create( fields={field: int(records[field])})
                        fields_to_update.append(field)
                        field_type.append('int')
                    except:
                        api.create( fields={field: float(records[field])})
                        fields_to_update.append(field)
                        field_type.append('float')
            elif 'Unknown field name' in str(e):
                print(field, " does not exist in Airtable, please create if you want to add")
            elif 'NVALID_ATTACHMENT_OBJECT' in str(e):
                fields_to_update.append(field)
                field_type.append('image')
                
            else:
                print(field, e)
    return dict(zip(fields_to_update,field_type)),fields_to_update

def convert_columns_to_needed_format(df,dict_fields_conversion):
    for keys, values in dict_fields_conversion.items():
        if values == 'str':
            df[keys] = df[keys].astype(str)
        elif values == 'int':
            df[keys] = df[keys].astype(int)
        elif values == 'float':
            df[keys] = df[keys].astype(str)
        elif values == 'image':
            df[keys].map(prepare_image_field_to_upload)
    return df

def prepare_image_field_to_upload(x):
    list_to_pop = ['height','id','width','filename','size','type','thumbnails']
    if type(x) == list:
        try:
            for image in x:
                for pops in list_to_pop:
                    try:
                        image.pop(pops)
                    except:
                        continue
                
        except:
            1
    else:
        1
    return x

def prepare_data_Airtable(df,columns_to_keep=False):
    a , b = columns_to_include_and_conversion_type(df)
    if columns_to_keep:
        return convert_columns_to_needed_format(df,a)[b+columns_to_keep]
    else:
        return convert_columns_to_needed_format(df,a)[b]