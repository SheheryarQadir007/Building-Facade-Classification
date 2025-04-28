import pandas as pd
from airtable import Airtable
import os
from unidecode import unidecode
from airtable_download_upload import airtable_download_upload
class DataFrame:
    def __init__(self, api, base_name):
        self.api = api
        self.base_name = base_name

    def download_df_fincas_nameproperly(self):
        table = self.api.table(self.base_name, 'Fincas')
        columns_to_include = [['FINCA', 'Año construcción', 'Tipo Finca', 'Parcela Catastral', 'parcela_catastral_joinkey', 'Codigo Postal', 'Address Validated AI']]
        formula = "{parcela_catastral_joinkey}"
        table = table.all(formula=formula, fields=columns_to_include)
        df_fincas_toSplit = pd.DataFrame(table)
        lista_records = list(df_fincas_toSplit['id'])
        df_fincas_toSplit = pd.DataFrame(list(df_fincas_toSplit['fields']))
        df_fincas_toSplit['record_id'] = lista_records
        df_fincas_toSplit['Codigo Postal'] = df_fincas_toSplit['Codigo Postal'].apply(lambda x: int(x) if pd.notna(x) else None).fillna('')
        df_fincas_toSplit['Finca_Proper_google'] = df_fincas_toSplit['FINCA'].str.title() + ', Madrid'
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['parcela_catastral_joinkey'].notna()]
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['Address Validated AI'].notna()]
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['Finca_Proper_google'].notna()]
        df_fincas_toSplit['ascii_filename'] = df_fincas_toSplit['Address Validated AI'].map(self.ascii_filename)

        dict_tipo_fincas = {
            'Representativa +5%': 'Clasica',
            'Clásica +0%': 'Clasica',
            'Moderna -10%': 'Moderna',
            'Moderna-Clásica -5%': 'Moderna',
            'Asintónica -20%': 'Moderna'
        }

        df_fincas_toSplit['Tipo Finca'] = df_fincas_toSplit['Tipo Finca'].replace(dict_tipo_fincas)
        return df_fincas_toSplit

    def ascii_filename(self, file_name):
        # Convert non-ASCII characters to their closest ASCII counterparts
        name, extension = os.path.splitext(file_name)
        ascii_name = unidecode(name)
        # Replace spaces with underscores and remove non-alphanumeric characters
        ascii_name = ''.join(e for e in ascii_name if e.isalnum() or e in ('-', '_')).rstrip()
        return f"{ascii_name}{extension}"
    

    def update_validated_address_AI(self, city='Madrid',airtable_download_upload=airtable_download_upload):
        table = self.api.table(self.base_name, 'Fincas')
        columns_to_include = ['Codigo Postal', 'FINCA']
        formula = "{Address Validated AI}=BLANK()"
        table = table.all(formula=formula, fields=columns_to_include)
        df_fincas = pd.DataFrame(table)

        try:
            lista_records = list(df_fincas['id'])
            df_fincas = pd.DataFrame(list(df_fincas['fields']))
            df_fincas['record_id'] = lista_records
            df_fincas['Codigo Postal'] = df_fincas['Codigo Postal'].apply(lambda x: int(x) if pd.notna(x) else None).fillna('')
            df_fincas['Finca_Proper_google'] = df_fincas['FINCA'].str.title() + ', ' + df_fincas['Codigo Postal'].astype(str) + ', ' + city
            df_fincas['Address_Validated'] = df_fincas['Finca_Proper_google'].apply(lambda x: self.validate_address(x, city))
            df_fincas['Codigo Postal'] = df_fincas['Address_Validated'].apply(self.extract_postal_code)
            df_fincas['Address Validated AI'] = df_fincas['Address_Validated']
            airtable_download_upload.batch_update_airtable(df_fincas, ['Codigo Postal', 'Address Validated AI'], 'Fincas', start_with=0)
            return 'Success!'
        except:
            return df_fincas
        
