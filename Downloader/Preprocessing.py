import os
import pandas as pd 
import requests
from google.cloud import storage,vision
from urllib.parse import urlparse
from pyairtable import Api, Base, Table
from geopy.geocoders import Nominatim
from unidecode import unidecode
import googlemaps
import time
import re
import airtable_download_upload
from io import BytesIO
import logging
from PIL import Image
import numpy as np   
from dataFrame import DataFrame
from airtable_download_upload import AirtableDownloadUpload
from  downloader import ImageDownloader
from geopy.geocoders import Nominatim
from pyairtable import Api
from utils import Utilities
class Preprocessing:
    def __init__(self, api, base_name, google_maps_api_key, bucket_name, api_key, user_agent):
        self.api = api
        self.base_name = base_name
        self.data_frame = DataFrame(api, base_name)
        self.image_downloader = ImageDownloader(google_maps_api_key, base_name, bucket_name, api_key, user_agent)
        self.utils = Utilities(google_maps_api_key)


    def separate_paths(self, paths):
            catastro_path = None
            other_path = None

            for path in paths:
                if path.endswith("catastro.jpeg") or path.endswith("catastro.jpg"):
                    catastro_path = path
                else:
                    other_path = path

            return pd.Series([catastro_path, other_path])
        
    def find_corrupted_images(self, catastro_path, other_path):
        """
        Scans multiple directories recursively to find corrupted .jpg and .jpeg images.

        Args:
            catastro_path (str): Path to the Catastro images directory.
            other_path (str): Path to the other images directory.

        Returns:
            list: A list of file paths that are corrupted.
        """
        # Initialize an empty list to store paths of corrupted files
        corrupted_files = []

        # List of directories to scan
        directories = [catastro_path, other_path]

        # Iterate over each directory provided
        for directory in directories:
            logging.info(f"Scanning directory for corrupted images: {directory}")

            # Walk through the directory recursively
            for root, _, files in os.walk(directory):
                for file in files:
                    # Check if the file has a .jpg or .jpeg extension (case-insensitive)
                    if file.lower().endswith((".jpg", ".jpeg")):
                        filepath = os.path.join(root, file)
                        try:
                            # Attempt to open and verify the image
                            with Image.open(filepath) as img:
                                img.verify()  # Verify if the image is not corrupted
                        except (IOError, SyntaxError) as e:
                            # Log a warning if the image is corrupted
                            logging.warning(f"Corrupted image found: {filepath} - {e}")
                            # Add the corrupted file path to the list
                            corrupted_files.append(filepath)

        # Return the list of corrupted file paths
        return corrupted_files



    def remove_corrupted_images(self, corrupted_files):
        """
        Removes corrupted image files from the filesystem.

        Args:
            corrupted_files (list): List of file paths that are corrupted.

        Returns:
            int: The number of files successfully removed.
        """
        removed_count = 0
        for filepath in corrupted_files:
            try:
                os.remove(filepath)
                logging.info(f"Removed corrupted image: {filepath}")
                removed_count += 1
            except Exception as e:
                logging.error(f"Failed to remove {filepath}. Error: {e}")
        return removed_count

    def resize_images(self, catastro_path, other_path) -> dict:
            # Initialize a dictionary to store output directories
            resized_dirs = {
                "resized_catastro": "",
                "resized_other": ""
            }
        
            # Define the output directories
            resized_catastro_dir = os.path.join(os.path.dirname(catastro_path), "resized_catastro")
            resized_other_dir = os.path.join(os.path.dirname(other_path), "resized_other")
        
            # Create the output directories if they don't exist
            os.makedirs(resized_catastro_dir, exist_ok=True)
            os.makedirs(resized_other_dir, exist_ok=True)
        
            # Update the resized_dirs dictionary
            resized_dirs["resized_catastro"] = resized_catastro_dir
            resized_dirs["resized_other"] = resized_other_dir

    def convert_images_to_jpeg(
            self,
            resized_catastro_dir: str,
            resized_other_dir: str,
            output_catastro_dir: str,
            output_other_dir: str
        ) -> tuple:
            """
            Converts all .png, .jpg, and .jpeg images in both resized_catastro_dir and resized_other_dir
            to .jpeg format. Saves the converted images in the specified output directories.
            Removes the original files if they were converted to .jpeg.

            Args:
                resized_catastro_dir (str): Path to the resized Catastro images directory.
                resized_other_dir (str): Path to the resized Other images directory.
                output_catastro_dir (str): Directory to save converted Catastro images.
                output_other_dir (str): Directory to save converted Other images.

            Returns:
                tuple: A tuple containing paths to the converted directories.
                    Example:
                    (
                        "path/to/converted_catastro",
                        "path/to/converted_other"
                    )
            """
            # Define the output directories
            converted_catastro_dir = output_catastro_dir
            converted_other_dir = output_other_dir

            # Create the output directories if they don't exist
            os.makedirs(converted_catastro_dir, exist_ok=True)
            os.makedirs(converted_other_dir, exist_ok=True)

            # Define a helper function to process and convert images in a given directory
            def process_directory(input_dir: str, output_dir: str):
                """
                Processes and converts images from input_dir and saves them to output_dir.

                Args:
                    input_dir (str): Path to the input directory containing images.
                    output_dir (str): Path to the output directory to save converted images.
                """
                # Check if the input directory exists
                if not os.path.isdir(input_dir):
                    logging.error(f"Input directory does not exist: {input_dir}")
                    return

                # Walk through the directory recursively
                for root, _, files in os.walk(input_dir):
                    for file in files:
                        # Check for .png, .jpg, and .jpeg files (case-insensitive)
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            input_filepath = os.path.join(root, file)
                            relative_path = os.path.relpath(root, input_dir)
                            output_subdir = os.path.join(output_dir, relative_path)
                            os.makedirs(output_subdir, exist_ok=True)
                            output_file_name = os.path.splitext(file)[0] + '.jpeg'
                            output_filepath = os.path.join(output_subdir, output_file_name)

                            try:
                                with Image.open(input_filepath) as img:
                                    # Convert image to RGB (necessary for JPEG format)
                                    rgb_img = img.convert('RGB')
                                    # Save the image in JPEG format
                                    rgb_img.save(output_filepath, 'JPEG')
                                    logging.info(f"Converted and saved image: {output_filepath}")

                                # Remove the original file if it's not already a .jpeg
                                if file.lower().endswith(('.png', '.jpg')) and output_filepath != input_filepath:
                                    os.remove(input_filepath)
                                    logging.info(f"Removed original image: {input_filepath}")

                            except Exception as e:
                                logging.error(f"Failed to convert image {input_filepath}: {e}")

            # Process the Catastro directory
            logging.info(f"Converting images in Catastro directory: {resized_catastro_dir}")
            process_directory(resized_catastro_dir, converted_catastro_dir)

            # Process the Other directory
            logging.info(f"Converting images in Other directory: {resized_other_dir}")
            process_directory(resized_other_dir, converted_other_dir)

            # Return the paths to the converted directories as a tuple
            return converted_catastro_dir, converted_other_dir
    
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
    
    def download_photos_save_in_folders(self, df_fincas_toSplit):

        fincas_Novaloradas = df_fincas_toSplit[~df_fincas_toSplit['Tipo Finca'].isin(['Clasica', 'Moderna'])]
        fincas_Novaloradas['etiqueta'] = 'no_valorada'

        try:
            no_valoradas_parcela_catastral_joinkey_evaluated_and_saved = self.load_list_with_numpy('no_valoradas_parcela_catastral_joinkey_evaluated_and_saved')
        except:
            no_valoradas_parcela_catastral_joinkey_evaluated_and_saved = []

        for etiqueta in fincas_Novaloradas['etiqueta'].unique():
            tipo_finca_df = fincas_Novaloradas[fincas_Novaloradas['etiqueta'] == etiqueta]
            i = 0
            for parcela_catastral_joinkey in tipo_finca_df.parcela_catastral_joinkey.unique():
                if parcela_catastral_joinkey not in no_valoradas_parcela_catastral_joinkey_evaluated_and_saved:
                    print(i, end='\r')
                    if i >= 0:
                        try:
                            self.image_downloader.finca_get_street_maps_photo_download(fincas_Novaloradas, etiqueta, parcela_catastral_joinkey, 'fincas_NoValoradas/')
                            no_valoradas_parcela_catastral_joinkey_evaluated_and_saved.append(parcela_catastral_joinkey)
                            self.save_list_fast(no_valoradas_parcela_catastral_joinkey_evaluated_and_saved, 'no_valoradas_parcela_catastral_joinkey_evaluated_and_saved.txt')
                            i += 1
                        except:
                            i += 1
                            pass
                    else:
                        i += 1

        fincas_valoradas = df_fincas_toSplit[df_fincas_toSplit['Tipo Finca'].isin(['Clasica', 'Moderna'])]
        fincas_valoradas.loc[fincas_valoradas['Tipo Finca'] == 'Moderna', 'etiqueta'] = 'noclasica'
        fincas_valoradas.loc[fincas_valoradas['Tipo Finca'] == 'Clasica', 'etiqueta'] = 'clasica'

        try:
            valoradas_parcela_catastral_joinkey_evaluated_and_saved = self.utils.load_list_with_numpy('valoradas_parcela_catastral_joinkey_evaluated_and_saved')
        except:
            valoradas_parcela_catastral_joinkey_evaluated_and_saved = []

        for etiqueta in fincas_valoradas['etiqueta'].unique():
            tipo_finca_df = fincas_valoradas[fincas_valoradas['etiqueta'] == etiqueta]
            i = 0
            for parcela_catastral_joinkey in tipo_finca_df.parcela_catastral_joinkey.unique():
                if parcela_catastral_joinkey not in valoradas_parcela_catastral_joinkey_evaluated_and_saved:
                    print(i, end='\r')
                    if i >= 0:
                        try:
                            self.image_downloader.finca_get_street_maps_photo_download(fincas_valoradas, etiqueta, parcela_catastral_joinkey, 'fotos_fincas/')
                            valoradas_parcela_catastral_joinkey_evaluated_and_saved.append(parcela_catastral_joinkey)
                            self.save_list_fast(valoradas_parcela_catastral_joinkey_evaluated_and_saved, 'valoradas_parcela_catastral_joinkey_evaluated_and_saved.txt')
                            i += 1
                        except:
                            i += 1
                            pass
                    else:
                        i += 1




    def download_valoradas_photos(self, df_fincas_toSplit: pd.DataFrame, folder: str):
        """
        Downloads photos for valoradas (valued) fincas ('Clasica' and 'Moderna').

        This method:
            1. Filters the DataFrame for 'Clasica' and 'Moderna' fincas.
            2. Labels them as 'clasica' and 'noclasica'.
            3. Loads a list of already processed parcela_catastral_joinkey to avoid duplicate processing.
            4. Iterates through each unique parcela_catastral_joinkey and downloads photos if not already processed.
            5. Updates the processed list and saves it after each successful download.

        Args:
            df_fincas_toSplit (pd.DataFrame): DataFrame containing fincas information.
            folder (str): Local folder to save downloaded images.
        """
       
    # Step 1: Filter for 'Clasica' and 'Moderna' fincas
        fincas_valoradas = df_fincas_toSplit[df_fincas_toSplit['Tipo Finca'].isin(['Clasica', 'Moderna'])].copy()

        # Step 2: Label them as 'clasica' and 'noclasica'
        fincas_valoradas.loc[fincas_valoradas['Tipo Finca'] == 'Moderna', 'etiqueta'] = 'noclasica'
        fincas_valoradas.loc[fincas_valoradas['Tipo Finca'] == 'Clasica', 'etiqueta'] = 'clasica'

        # Step 3: Load the list of already processed parcela_catastral_joinkey
        try:
            valoradas_parcela_catastral_joinkey_evaluated_and_saved = self.utils.load_list_with_numpy(
                'valoradas_parcela_catastral_joinkey_evaluated_and_saved.json'
            )
        except Exception as e:
            logging.error(f"Error loading processed keys: {e}")
            valoradas_parcela_catastral_joinkey_evaluated_and_saved = []

        # Step 4: Iterate through each unique etiqueta ('clasica', 'noclasica')
        for etiqueta in fincas_valoradas['etiqueta'].unique():
            tipo_finca_df = fincas_valoradas[fincas_valoradas['etiqueta'] == etiqueta]
            for i, parcela_catastral_joinkey in enumerate(tipo_finca_df.parcela_catastral_joinkey.unique(), start=1):
                if parcela_catastral_joinkey not in valoradas_parcela_catastral_joinkey_evaluated_and_saved:
                    logging.info(f"Processing {i}: {parcela_catastral_joinkey} as {etiqueta}")
                    try:
                        self.image_downloader.finca_get_street_maps_photo_download(
                            df=fincas_valoradas,
                            tipo=etiqueta,
                            location=parcela_catastral_joinkey,
                            folder=folder
                        )
                        # Append the processed key to the list
                        valoradas_parcela_catastral_joinkey_evaluated_and_saved.append(parcela_catastral_joinkey)
                        # Save the updated list to the file
                        self.utils.save_list_fast(
                            valoradas_parcela_catastral_joinkey_evaluated_and_saved,
                            'valoradas_parcela_catastral_joinkey_evaluated_and_saved.json'
                        )
                        logging.info(f"Successfully processed {parcela_catastral_joinkey}")
                    except Exception as e:
                        logging.error(f"Failed to download photo for {parcela_catastral_joinkey}: {e}")
                        continue    
    def NOvaloradas_download_photo_saveinfolder(self, df):
            fincas_Novaloradas = df[~df['Tipo Finca'].isin(['Clasica', 'Moderna'])]
            fincas_Novaloradas['etiqueta'] = 'no_valorada'

            for etiqueta in fincas_Novaloradas['etiqueta'].unique():
                tipo_finca_df = fincas_Novaloradas[fincas_Novaloradas['etiqueta'] == etiqueta]
                i = 0
                for parcela_catastral_joinkey in tipo_finca_df.parcela_catastral_joinkey.unique():
                    print(i, end='\r')
                    if i >= 0:
                        try:
                            self.finca_get_street_maps_photo_download(i, etiqueta, parcela_catastral_joinkey, 'fincas_NoValoradas/')
                            i += 1
                        except:
                            i += 1
                            pass
                    else:
                        i += 1
    def run_dict_fotos_fincas(self, df_fincas_toSplit):
            directories = [
                "fotos_fincas/clasica",
                "fotos_fincas/noclasica",
                "fincas_NoValoradas/no_valorada"
            ]

            dir_files = {dir: set(os.listdir(dir)) for dir in directories}
            dict_fotos_fincas = {}

            for parcela_catastral_joinkey in df_fincas_toSplit.parcela_catastral_joinkey.unique():
                record_id = df_fincas_toSplit[df_fincas_toSplit['parcela_catastral_joinkey'] == parcela_catastral_joinkey]['record_id'].unique()[0]
                list_files_fincas = []

                filenames = [
                    parcela_catastral_joinkey + 'catastro.jpg',
                    parcela_catastral_joinkey + '.jpg'
                ]

                for dir in directories:
                    for fname in filenames:
                        if fname in dir_files[dir]:
                            list_files_fincas.append(f"{dir}/{fname}")

                dict_fotos_fincas[record_id] = list_files_fincas
            return dict_fotos_fincas
