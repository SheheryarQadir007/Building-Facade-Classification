#libraries 

import datetime
from email.mime import image
import os
import sys
import importlib
from IPython.display import clear_output
import requests
import time
import logging
# Google Maps Paid API
from geopy.geocoders import GoogleV3
from google.cloud import vision, secretmanager, storage
from geopy.geocoders import Nominatim
import googlemaps
from pyairtable import Api, Base, Table
import pandas as pd 
import re
import numpy as np
import json
from io import BytesIO
from urllib.parse import quote
from unidecode import unidecode
from PIL import Image
from pathlib import Path
import platform
from ultralytics import YOLO
import cv2
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# sys.path.append("/Users/luismdecarvajal/Library/CloudStorage/GoogleDrive-l.carvajal@atipikproperties.com/My Drive/functions/airtable/")
import airtable_download_upload
from unidecode import unidecode

# storage_client = storage.Client()
# bucket = storage_client.bucket(BUCKET_NAME)


class FincaPhotoDownloader:
    def __init__(self, google_maps_api_key, api_key, base_name, bucket_name, user_agent, modelyolo, cmodel):
        self.google_maps_api_key = google_maps_api_key
        self.api = Api(api_key)
        self.base_name = base_name
        self.bucket_name = bucket_name
        self.geolocator = Nominatim(user_agent=user_agent)
        self.gmaps = googlemaps.Client(key=google_maps_api_key)

        self.IMG_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        self.INITIAL_EPOCHS = 50
        self.FINE_TUNE_EPOCHS = 10

        self.model = modelyolo  # YOLOv8 model for segmentation
        self.classification_model = cmodel 
        self.config_file = "training_config.json"
        self.processed_log = "processed_photos.log"

        # Define augmentations for training and validation
        self.train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5), 
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
            ], p=0.5),
            A.GridDropout(ratio=0.2, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize()
        ])

        self.val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize()
        ])
# step 1 : Arrange the data
    def finca_get_street_maps_photo_download(self, df, tipo, location, folder):
        meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
        pic_base = 'https://maps.googleapis.com/maps/api/streetview?'

        parcela_catastral_joinkey = location
        if tipo == 'no_valorada':
            par_catastral = df[df['parcela_catastral_joinkey'] == location]['Parcela Catastral'].unique()[0]
            location = df[df['parcela_catastral_joinkey'] == location]['Address Validated AI'].unique()[0]
        else:
            par_catastral = df[df['parcela_catastral_joinkey'] == location]['Parcela Catastral'].unique()[0]
            location = df[df['parcela_catastral_joinkey'] == location]['Address Validated AI'].unique()[0]

        meta_params = {'key': self.google_maps_api_key, 'location': location}
        pic_params = {'key': self.google_maps_api_key,
                    'location': location,
                    'size': "640x640",
                    'pitch': '30',
                    'source': 'outdoor'}

        # Check if the image is already downloaded and saved
        already_processed = self.load_list_with_numpy('no_valoradas_parcela_catastral_joinkey_evaluated_and_saved')
        if parcela_catastral_joinkey in already_processed:
            print(f"Image for {parcela_catastral_joinkey} already processed.")
            return

        try:
            pic_response = requests.get(pic_base, params=pic_params)
        except:
            time.sleep(120)
            pic_response = requests.get(pic_base, params=pic_params)

        # Check if the image exists in the cloud storage bucket before uploading
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)
        blob = bucket.blob(f'fotos_fincas/{tipo}/{parcela_catastral_joinkey}.jpg')

        if blob.exists():
            print(f"Image {parcela_catastral_joinkey} already exists in the bucket. Skipping upload.")
        else:
            try:
                response = client.label_detection(image=vision.Image(content=pic_response.content))
                labels = response.label_annotations
                for label in labels:
                    if label.description == 'Building' and label.score >= 0.7:
                        save_path = os.path.join(folder, tipo)
                        complete_name = os.path.join(save_path, parcela_catastral_joinkey)
                        with open(complete_name + '.jpg', 'wb') as file:
                            file.write(pic_response.content)

                # Upload the image to the cloud bucket
                blob.upload_from_file(BytesIO(pic_response.content), content_type='image/jpeg')
                print(f"Uploaded image {parcela_catastral_joinkey} to GCP.")
            except:
                save_path = os.path.join(folder, tipo)
                complete_name = os.path.join(save_path, parcela_catastral_joinkey)
                with open(complete_name + '.jpg', 'wb') as file:
                    file.write(pic_response.content)

            pic_response.close()
            already_processed.append(parcela_catastral_joinkey)
            self.save_list_fast(already_processed, 'no_valoradas_parcela_catastral_joinkey_evaluated_and_saved')

    # Handle the Catastro image similarly, if needed
        if type(par_catastral) is not float:
            url = f'http://ovc.catastro.meh.es/OVCServWeb/OVCWcfLibres/OVCFotoFachada.svc/RecuperarFotoFachadaGet?ReferenciaCatastral={par_catastral}'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, headers=headers)
            image = vision.Image(content=response.content)

            try:
                response = client.label_detection(image=image)
                labels = response.label_annotations
                for label in labels:
                    if label.description == 'Building' and label.score >= 0.7:
                        save_path = os.path.join(folder, tipo)
                        complete_name = os.path.join(save_path, parcela_catastral_joinkey)
                        with open(complete_name + 'catastro.jpg', 'wb') as file:
                            file.write(response.content)

                # Upload the Catastro image to the cloud bucket
                blob = bucket.blob(f'fotos_fincas/{tipo}/{parcela_catastral_joinkey}catastro.jpg')
                if not blob.exists():
                    blob.upload_from_file(BytesIO(response.content), content_type='image/jpeg')
                    print(f"Uploaded Catastro image {parcela_catastral_joinkey} to GCP.")
            except:
                save_path = os.path.join(folder, tipo)
                complete_name = os.path.join(save_path, parcela_catastral_joinkey)
                with open(complete_name + 'catastro.jpg', 'wb') as file:
                    file.write(response.content)

            response.close()
    
    
    def validate_address(self, address, city):
        try:
            location = self.gmaps.addressvalidation([address], regionCode='ES', locality=city)['result']['address']['formattedAddress']
            return location
        except:
            return 'Error'

    def extract_postal_code(self, text):
        try:
            numbers = re.findall(r'\d+', text)
            large_numbers = [int(num) for num in numbers if int(num) > 1000]
            return str(int(large_numbers[0]))
        except:
            return None

    def update_validated_address_AI(self, city='Madrid'):
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
        




    def ascii_filename(self, file_name):
        # Convert non-ASCII characters to their closest ASCII counterparts
        name, extension = os.path.splitext(file_name)
        ascii_name = unidecode(name)
        # Replace spaces with underscores and remove non-alphanumeric characters
        ascii_name = ''.join(e for e in ascii_name if e.isalnum() or e in ('-', '_')).rstrip()
        return f"{ascii_name}{extension}"

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

    def valoradas_download_photo_saveinfolder(self, df):
        fincas_valoradas = df[df['Tipo Finca'].isin(['Clasica', 'Moderna'])]

        fincas_valoradas.loc[df['Tipo Finca'] == 'Moderna', 'etiqueta'] = 'noclasica'
        fincas_valoradas.loc[df['Tipo Finca'] == 'Clasica', 'etiqueta'] = 'clasica'

        for etiqueta in fincas_valoradas['etiqueta'].unique():
            tipo_finca_df = fincas_valoradas[fincas_valoradas['etiqueta'] == etiqueta]
            i = 0
            for parcela_catastral_joinkey in tipo_finca_df.parcela_catastral_joinkey.unique():
                print(i, end='\r')
                if i >= 0:
                    try:
                        self.finca_get_street_maps_photo_download(i, etiqueta, parcela_catastral_joinkey, 'fotos_fincas/')
                        i += 1
                    except:
                        i += 1
                        pass
                else:
                    i += 1

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

    def generate_df_streetmaps_catastro_paths(self, dict_fotos_fincas):
        df_fincas_valoradas = pd.DataFrame()

        record_id_list = []
        paths_list = []

        for record_id, paths in dict_fotos_fincas.items():
            record_id_list.append(record_id)
            paths_list.append(paths)

        df_fincas_valoradas['record_id'] = record_id_list
        df_fincas_valoradas['paths'] = paths_list

        df_fincas_valoradas[['catastro_path', 'streetmaps_path']] = df_fincas_valoradas['paths'].apply(self.separate_paths)
        return df_fincas_valoradas

#Step 2: prepare the data

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
    
    def process_no_valoradas(self):
        df_fincas = self.update_validated_address_AI()
        print(df_fincas)
        df_fincas_toSplit = self.download_df_fincas_nameproperly()

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
                            self.finca_get_street_maps_photo_download(fincas_Novaloradas, etiqueta, parcela_catastral_joinkey, 'fincas_NoValoradas/')
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
            valoradas_parcela_catastral_joinkey_evaluated_and_saved = self.load_list_with_numpy('valoradas_parcela_catastral_joinkey_evaluated_and_saved')
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
                            self.finca_get_street_maps_photo_download(fincas_valoradas, etiqueta, parcela_catastral_joinkey, 'fotos_fincas/')
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
        df_fincas = self.update_validated_address_AI()
        print(df_fincas)
        df_fincas_toSplit = self.download_df_fincas_nameproperly()
    # Step 1: Filter for 'Clasica' and 'Moderna' fincas
        fincas_valoradas = df_fincas_toSplit[df_fincas_toSplit['Tipo Finca'].isin(['Clasica', 'Moderna'])].copy()

        # Step 2: Label them as 'clasica' and 'noclasica'
        fincas_valoradas.loc[fincas_valoradas['Tipo Finca'] == 'Moderna', 'etiqueta'] = 'noclasica'
        fincas_valoradas.loc[fincas_valoradas['Tipo Finca'] == 'Clasica', 'etiqueta'] = 'clasica'

        # Step 3: Load the list of already processed parcela_catastral_joinkey
        try:
            valoradas_parcela_catastral_joinkey_evaluated_and_saved = self.load_list_with_numpy(
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
                        self.finca_get_street_maps_photo_download(
                            df=fincas_valoradas,
                            tipo=etiqueta,
                            location=parcela_catastral_joinkey,
                            folder=folder
                        )
                        # Append the processed key to the list
                        valoradas_parcela_catastral_joinkey_evaluated_and_saved.append(parcela_catastral_joinkey)
                        # Save the updated list to the file
                        self.save_list_fast(
                            valoradas_parcela_catastral_joinkey_evaluated_and_saved,
                            'valoradas_parcela_catastral_joinkey_evaluated_and_saved.json'
                        )
                        logging.info(f"Successfully processed {parcela_catastral_joinkey}")
                    except Exception as e:
                        logging.error(f"Failed to download photo for {parcela_catastral_joinkey}: {e}")
                        continue    


    def save_list_fast(self, my_list, file_path):
        with open(file_path, 'w') as f:
            f.write('\n'.join(my_list))

    def load_list_with_numpy(self, file_name):
        return np.loadtxt(f"{file_name}.txt", dtype=str).tolist()

#Step 3: Segmemtation the data
    def Segmentimage(image_path: str,model: YOLO,target_label: str = "main_building") -> np.ndarray:
        """
        Processes a single image by detecting the target label using the YOLO model,
        cropping the image to the bounding box of the most confident target label,
        and returning the cropped image. If the target label is not detected,
        returns the original image.

        Args:
            image_path (str): Path to the input image.
            model (YOLO): Loaded YOLO model for prediction.
            target_label (str, optional): The label to detect and crop. Defaults to "main_building".

        Returns:
            np.ndarray: The cropped image if target label is detected, else the original image.
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None

        # Convert to RGB for consistency with model predictions
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        try:
            results = model.predict(source=image_path, save=False, imgsz=640, iou=0.5)
        except Exception as e:
            logging.error(f"Error during model prediction: {e}")
            return image_rgb  # Return original image on prediction failure

        # Initialize variables to keep track of the best bounding box
        best_confidence = 0
        best_box = None

        # Iterate over the results to find the best bounding box for the target label
        for result in results:
            for box, confidence, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label_name = model.names[int(cls)]
                if label_name == target_label and confidence > best_confidence:
                    best_confidence = confidence
                    best_box = box.cpu().numpy().astype(int)  # Convert to integer coordinates

        if best_box is not None:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = best_box

            # Validate coordinates to be within image bounds
            height, width, _ = image_rgb.shape
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                logging.warning(f"Invalid bounding box for image {image_path}: {best_box}")
                return image_rgb  # Return original image if bounding box is invalid

            # Crop the image to the bounding box
            cropped_image = image_rgb[y1:y2, x1:x2]

            logging.info(f"Cropped image {image_path} to bounding box: {best_box} with confidence: {best_confidence}")

            return cropped_image
        else:
            logging.info(f"No '{target_label}' detected in image: {image_path}, returning original image.")
            return image_rgb


    def apply_segmentation_to_directory(directory: str, model: YOLO, target_label: str = "main_building"):
        """
        Applies segmentation to all JPEG images in the given directory (and subdirectories)
        using a YOLO model and crops the images to the bounding box of the target label.

        Args:
            directory (str): Path to the base directory containing images to process.
            model (YOLO): Loaded YOLO model for segmentation.
            target_label (str, optional): The label to detect and crop. Defaults to "main_building".

        Returns:
            None: Processes images in place.
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Iterate over all JPEG files in the directory and subdirectories
        image_files = list(Path(directory).rglob("*.jpeg"))  # Only process .jpeg files

        for image_path in image_files:
            try:
                # Convert image path to string
                image_path_str = str(image_path)

                # Load the image
                image = cv2.imread(image_path_str)
                if image is None:
                    logging.warning(f"Failed to load image: {image_path}")
                    continue

                # Convert to RGB for consistency with YOLO
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Run segmentation
                results = model.predict(source=image_path_str, save=False, imgsz=640, iou=0.5)

                # Find the best bounding box for the target label
                best_confidence = 0
                best_box = None
                for result in results:
                    for box, confidence, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        label_name = model.names[int(cls)]
                        if label_name == target_label and confidence > best_confidence:
                            best_confidence = confidence
                            best_box = box.cpu().numpy().astype(int)  # Convert to integer coordinates

                if best_box is not None:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = best_box

                    # Validate coordinates
                    height, width, _ = image.shape
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)

                    if x2 > x1 and y2 > y1:
                        # Crop the image
                        cropped_image = image_rgb[y1:y2, x1:x2]

                        # Save the cropped image back to the same path in JPEG format
                        cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(image_path_str, cropped_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        logging.info(f"Segmented and saved image: {image_path}")
                    else:
                        logging.warning(f"Invalid bounding box for {image_path}: {best_box}")
                else:
                    logging.info(f"No '{target_label}' detected in {image_path}. Skipping segmentation.")

            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")




    def predict_image(path, img_height, img_width, model, class_names):
        try:
            img = tf.keras.utils.load_img('./' + path, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            confidence = 100 * np.max(score)
            classification = class_names[np.argmax(score)]
            return classification, confidence
        except:
            return 'UnidentifiedImageError', 'UnidentifiedImageError'




    def apply_model_generate_df_classification_confidence(self, df_fincas_valoradas, classification_model, segmentation_model, img_height, img_width, class_names, target_label="main_building"):
        """
        Processes the dataframe by segmenting each image, applying the classification model,
        and storing the results with confidence scores.

        Args:
            df_fincas_valoradas (pd.DataFrame): DataFrame containing paths to the images.
            classification_model: Pre-trained classification model.
            segmentation_model (YOLO): YOLO model for segmentation.
            img_height (int): Height to resize images for classification.
            img_width (int): Width to resize images for classification.
            class_names (list): List of class names for the classification model.
            target_label (str, optional): Target label for segmentation. Defaults to "main_building".

        Returns:
            pd.DataFrame: Updated DataFrame with classifications and confidence scores.
        """
        catastro_clasification_list = []
        catastro_confidence_list = []
        streetmaps_clasification_list = []
        streetmaps_confidence_list = []

        for _, row in df_fincas_valoradas.iterrows():
            # Process Catastro image
            if row.catastro_path != '':
                segmented_image = self.Segmentimage(row.catastro_path, segmentation_model, target_label)
                classification, confidence = self.predict_image(segmented_image, img_height, img_width, classification_model, class_names)
                catastro_clasification_list.append(classification)
                catastro_confidence_list.append(confidence)
            else:
                catastro_clasification_list.append(np.NaN)
                catastro_confidence_list.append(np.NaN)

            # Process Streetmaps image
            if row.streetmaps_path != '':
                segmented_image = self.Segmentimage(row.streetmaps_path, segmentation_model, target_label)
                classification, confidence = self.predict_image(segmented_image, img_height, img_width, classification_model, class_names)
                streetmaps_clasification_list.append(classification)
                streetmaps_confidence_list.append(confidence)
            else:
                streetmaps_clasification_list.append(np.NaN)
                streetmaps_confidence_list.append(np.NaN)

            clear_output(wait=True)

        # Update DataFrame
        df_fincas_valoradas['catastro_clasification'] = catastro_clasification_list
        df_fincas_valoradas['catastro_confidence'] = catastro_confidence_list
        df_fincas_valoradas['streetmaps_clasification'] = streetmaps_clasification_list
        df_fincas_valoradas['streetmaps_confidence'] = streetmaps_confidence_list

        return df_fincas_valoradas





    def generate_classification_confidence_df(self, df):
        df['catastro_confidence'] = pd.to_numeric(df['catastro_confidence'], errors='coerce')
        df['streetmaps_confidence'] = pd.to_numeric(df['streetmaps_confidence'], errors='coerce')

        # Asigna 'Clasica' si ambas clasificaciones son 'clasica' y una de las dos confianzas es mayor de 87 y la otra mayor de 70
        df.loc[(df.catastro_clasification == 'clasica') & (df.catastro_confidence >= 87) & (df.streetmaps_clasification == 'clasica') & (df.streetmaps_confidence >= 70), 'Tipo Finca AI'] = 'Clasica'
        df.loc[(df.catastro_clasification == 'clasica') & (df.catastro_confidence >= 70) & (df.streetmaps_clasification == 'clasica') & (df.streetmaps_confidence >= 87), 'Tipo Finca AI'] = 'Clasica'

        # Asigna 'Moderna' si ambas clasificaciones son 'noclasica' y una de las dos confianzas es mayor de 87 y la otra mayor de 70
        df.loc[(df.catastro_clasification == 'noclasica') & (df.catastro_confidence >= 87) & (df.streetmaps_clasification == 'noclasica') & (df.streetmaps_confidence >= 70), 'Tipo Finca AI'] = 'Moderna'
        df.loc[(df.catastro_clasification == 'noclasica') & (df.catastro_confidence >= 70) & (df.streetmaps_clasification == 'noclasica') & (df.streetmaps_confidence >= 87), 'Tipo Finca AI'] = 'Moderna'

        # Asigna clasica  si uno de los dos es mayor que 90 + el otro es nulo
        df.loc[(df.catastro_clasification == 'clasica') & (df.catastro_confidence >= 90) & pd.isna(df.streetmaps_confidence), 'Tipo Finca AI'] = 'Clasica'
        df.loc[pd.isna(df.catastro_confidence) & (df.streetmaps_clasification == 'clasica') & (df.streetmaps_confidence >= 90), 'Tipo Finca AI'] = 'Clasica'
        #si sabemos año , con 71 de confianza ya vale
        df.loc[(df['Año construcción'] <= 1950) & (df.catastro_clasification == 'clasica') & (df.catastro_confidence >= 71) & pd.isna(df.streetmaps_confidence), 'Tipo Finca AI'] = 'Clasica'
        df.loc[(df['Año construcción'] <= 1950) & (df.catastro_clasification == 'clasica') & (df.catastro_confidence >= 81) & (df.catastro_confidence >= 71) & (df.streetmaps_clasification == 'clasica') & (df.streetmaps_confidence >= 71), 'Tipo Finca AI'] = 'Clasica'

        # Asigna Moderna  si uno de los dos es mayor que 80 + el otro es nulo
        df.loc[(df.catastro_clasification == 'noclasica') & (df.catastro_confidence >= 80) & pd.isna(df.streetmaps_confidence), 'Tipo Finca AI'] = 'Moderna'
        df.loc[pd.isna(df.catastro_confidence) & (df.streetmaps_clasification == 'noclasica') & (df.streetmaps_confidence >= 80), 'Tipo Finca AI'] = 'Moderna'
        #si sabemos año , con 71 de confianza ya vale
        df.loc[(df['Año construcción'] > 1960) & (df.catastro_clasification == 'noclasica') & (df.catastro_confidence >= 60) & (df.streetmaps_clasification == 'noclasica') & (df.streetmaps_confidence >= 71), 'Tipo Finca AI'] = 'Moderna'
        df.loc[(df['Año construcción'] > 1960) & (df.catastro_clasification == 'noclasica') & (df.catastro_confidence >= 60) & (df.streetmaps_clasification == 'noclasica') & pd.isna(df.streetmaps_confidence), 'Tipo Finca AI'] = 'Moderna'

        # Moderna si ambas confianzas debajo de 50
        df.loc[(df.catastro_confidence <= 50) & (df.streetmaps_confidence <= 50), 'Tipo Finca AI'] = 'Moderna'

        # Moderna si ambas confianzas debajo de 70 y sabemos año
        df.loc[(df['Año construcción'] > 1960) & (df.catastro_confidence <= 70) & (df.streetmaps_confidence <= 70), 'Tipo Finca AI'] = 'Moderna'

        # Moderna si sabemos año y esta catalogado como no clasica por debajo de 80
        df.loc[(df['Año construcción'] > 1960) & (df.catastro_clasification == 'noclasica') & (df.catastro_confidence <= 80) & pd.isna(df.streetmaps_confidence), 'Tipo Finca AI'] = 'Moderna'

        # si no sabemos año
        df.loc[pd.isna(df.catastro_confidence) & (df.streetmaps_clasification == 'clasica') & (df.streetmaps_confidence <= 70), 'Tipo Finca AI'] = 'Moderna'

        # Moderna si sabemos año y esta catalogado como clasica por debajo de 60
        df.loc[(df['Año construcción'] > 1960) & (df.catastro_clasification == 'clasica') & (df.catastro_confidence <= 60) & pd.isna(df.streetmaps_confidence), 'Tipo Finca AI'] = 'Moderna'

        # si no sabemos año
        df.loc[pd.isna(df.catastro_confidence) & (df.streetmaps_clasification == 'noclasica') & (df.streetmaps_confidence <= 80), 'Tipo Finca AI'] = 'Moderna'

        df.loc[(pd.isna(df.catastro_clasification)) & (pd.isna(df.streetmaps_clasification)), 'Tipo Finca AI'] = 'NoImage'

        df.loc[(df['Año construcción'] <= 1930), 'Tipo Finca AI'] = 'Clasica'
        return df



   
    def process_directory(self, directory_path):
        image_files = list(Path(directory_path).glob('*/*.jpg')) + list(Path(directory_path).glob('*/*.png'))
        for image_file in image_files:
            try:
                # Resize and save image
                img = self.resize_images(image_file)
                img.save(image_file)
            except:
                os.remove(image_file)  # Remove invalid files

        # Rename files and convert to JPEG
        self.rename_files_in_directory(directory_path)
        self.convert_images_to_jpeg(directory_path)

  

  
  
  
    def rename_files_in_directory(self, directory_path):
        for foldername, _, filenames in os.walk(directory_path):
            for file_name in filenames:
                new_name = self.ascii_filename(file_name)
                if new_name != file_name:
                    os.rename(os.path.join(foldername, file_name), os.path.join(foldername, new_name))



    
    
    
    def validate_image_formats(self, data_dir):
        image_paths = list(data_dir.glob('*/*.jpg')) + list(data_dir.glob('*/*.png'))
        for path in image_paths:
            try:
                decoded_img = tf.image.decode_image(str(path))
                print(f"Successfully decoded {decoded_img}")
            except Exception as e:
                print(f"Failed to decode {decoded_img}. Error: {e}")

    # Main image processing function
    
    
    
    
    def process_images(self):
        directories = [
            './fotos_fincas/noclasica',
            './fotos_fincas/clasica',
            './fincas_NoValoradas/no_valorada'
        ]

        for directory in directories:
            self.process_directory(directory)

        # Validate image formats
        data_dir = Path('./fotos_fincas')
        self.validate_image_formats(data_dir)


    def organize_dataset(self, base_dir, output_dir):
        # Define paths for categories
        
        clasica_dir = os.path.join(base_dir, "clasica")
        noclasica_dir = os.path.join(base_dir, "noclasica")

        # Ensure the directories exist
        if not os.path.exists(clasica_dir):
            raise FileNotFoundError(f"Directory not found: {clasica_dir}")
        if not os.path.exists(noclasica_dir):
            raise FileNotFoundError(f"Directory not found: {noclasica_dir}")
        
        # Clear the output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Remove all contents of the output directory
            print(f"Cleared the output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)  # Recreate the directory structure

        # Get all image paths
        clasica_images = [os.path.join(clasica_dir, f) for f in os.listdir(clasica_dir) if f.endswith(('.jpg', '.png'))]
        noclasica_images = [os.path.join(noclasica_dir, f) for f in os.listdir(noclasica_dir) if f.endswith(('.jpg', '.png'))]

        # Create train-test-validation splits
        train_clasica, temp_clasica = train_test_split(clasica_images, test_size=0.3, random_state=42)
        val_clasica, test_clasica = train_test_split(temp_clasica, test_size=0.5, random_state=42)

        train_noclasica, temp_noclasica = train_test_split(noclasica_images, test_size=0.3, random_state=42)
        val_noclasica, test_noclasica = train_test_split(temp_noclasica, test_size=0.5, random_state=42)

        # Create output directories
        subsets = ["train", "validation", "test"]
        categories = ["clasica", "noclasica"]

        for subset in subsets:
            for category in categories:
                os.makedirs(os.path.join(output_dir, subset, category), exist_ok=True)

        # Helper function to copy images to the target directory
        def copy_images(image_list, target_dir):
            for img in image_list:
                shutil.copy(img, target_dir)

        # Organize the splits into the output directories
        copy_images(train_clasica, os.path.join(output_dir, "train", "clasica"))
        copy_images(val_clasica, os.path.join(output_dir, "validation", "clasica"))
        copy_images(test_clasica, os.path.join(output_dir, "test", "clasica"))

        copy_images(train_noclasica, os.path.join(output_dir, "train", "noclasica"))
        copy_images(val_noclasica, os.path.join(output_dir, "validation", "noclasica"))
        copy_images(test_noclasica, os.path.join(output_dir, "test", "noclasica"))
    


    def preprocess_image(self, img, transforms):
        """Apply augmentations to the image."""
        augmented = transforms(image=img)
        return augmented['image']

    def load_and_preprocess_img(self, img_path, transforms):
        """Load and apply transformations to the image from a path."""
        print(f"Loading image: {img_path}")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.preprocess_image(img, transforms)

    def create_data_generator(self, generator, transforms, batch_size=32):
        """
        Creates a custom data generator by applying augmentations to the image batches.

        :param generator: A generator (e.g., ImageDataGenerator) that yields batches of images and labels.
        :param transforms: Albumentations transformations to apply to the images.
        :param batch_size: The batch size for the data generator.
        :return: A data generator that applies transformations.
        """
        while True:  # Infinite loop for generator
            batch_images, batch_labels = next(generator)
            batch_images = np.array([self.preprocess_image(image, transforms) for image in batch_images])
            yield batch_images, batch_labels

    def get_last_training_date(self):
        """Reads the last training date from the config file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                return datetime.datetime.strptime(config.get("last_training_date"), "%Y-%m-%d")
        except FileNotFoundError:
            # If the config file does not exist, return None (meaning no training has occurred yet)
            return None

    def update_training_date(self):
        """Updates the config file with the current date after retraining."""
        with open(self.config_file, 'w') as f:
            config = {
                "last_training_date": datetime.datetime.today().strftime("%Y-%m-%d")
            }
            json.dump(config, f)

    def is_training_due(self):
        """Checks if the retraining is due based on the last training date."""
        last_training_date = self.get_last_training_date()
        
        if last_training_date is None:
            return True  # If no training has occurred yet, retraining is due
        
        # Calculate the difference between today's date and the last training date
        days_since_last_training = (datetime.datetime.today() - last_training_date).days
        
        # Check if more than 30 days have passed since the last training (one month)
        return days_since_last_training >= 30

    def fine_tune_model_monthly(self, train_dir, val_dir, epochs=10, batch_size=32, img_size=(224, 224)):
        """Fine-tunes the classification model with the new labeled data."""
        print("Fine-tuning the classification model with new data...")

        # Fine-tune the model using the updated dataset (train_dir and val_dir)
        self.train_and_evaluate_model(train_dir, val_dir, epochs, batch_size, img_size)

    def train_and_evaluate_model(self, train_dir, val_dir, epochs=50, batch_size=32, img_size=(224, 224)):
        """Train and evaluate the model on the provided dataset directories."""
        print(f"Training the model for {epochs} epochs...")

        # Initialize image data generators for training and validation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

        # Load the training and validation data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Build the MobileNetV2 model with a dropout layer
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze the base model initially

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Callbacks
        lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Train the model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[lr_reduction, tensorboard_callback]
        )

        # Save the fine-tuned model
        model.save("building_classification_model_finetuned.h5")
        print("Model fine-tuned and saved successfully.")
    
    def download_new_manually_tagged_images(self, df):
        directories = [
            "fotos_fincas/clasica",
            "fotos_fincas/noclasica"
        ]

        # Load processed photos log
        processed_photos = self.load_processed_photos()

        # Download and save new manually tagged images
        print("Downloading new manually tagged images...")
        self.download_and_save_photos(df, processed_photos)

    def download_and_save_photos(self, df, processed_photos):
        fincas_valoradas = df[df['Tipo Finca'].isin(['Clasica', 'Moderna'])]
        fincas_valoradas.loc[df['Tipo Finca'] == 'Moderna', 'etiqueta'] = 'noclasica'
        fincas_valoradas.loc[df['Tipo Finca'] == 'Clasica', 'etiqueta'] = 'clasica'

        for etiqueta, tipo_finca_df in [('clasica', fincas_valoradas[fincas_valoradas['etiqueta'] == 'clasica']),
                                        ('noclasica', fincas_valoradas[fincas_valoradas['etiqueta'] == 'noclasica'])]:
            i = 0
            for parcela_catastral_joinkey in tipo_finca_df.parcela_catastral_joinkey.unique():
                if parcela_catastral_joinkey in processed_photos:
                    continue  # Skip already processed photos
                print(i, end='\r')
                if i >= 0:
                    try:
                        self.download_photo(df, etiqueta, parcela_catastral_joinkey, f'fotos_fincas/{etiqueta}/')
                        self.log_processed_photo(parcela_catastral_joinkey)
                        i += 1
                    except:
                        i += 1
                        pass
                else:
                    i += 1
    def download_photo(self, df, tipo, location, folder):
        pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
        parcela_catastral_joinkey = location
        location = df[df['parcela_catastral_joinkey'] == location]['Address Validated AI'].unique()[0]

        pic_params = {'key': self.google_maps_api_key,
                      'location': location,
                      'size': "640x640",
                      'pitch': '30',
                      'source': 'outdoor'}

        try:
            pic_response = requests.get(pic_base, params=pic_params)
        except:
            time.sleep(120)
            pic_response = requests.get(pic_base, params=pic_params)

        save_path = os.path.join(folder, f"{parcela_catastral_joinkey}.jpg")
        with open(save_path, 'wb') as file:
            file.write(pic_response.content)
        print(f"Saved image {parcela_catastral_joinkey} to {save_path}")

    def load_processed_photos(self):
        if not os.path.exists(self.processed_log):
            open(self.processed_log, 'w').close()  # Create the file if it does not exist
            return set()
        with open(self.processed_log, "r") as file:
            return set(line.strip() for line in file)


    def log_processed_photo(self, parcela_catastral_joinkey):
        with open(self.processed_log, "a") as file:
            file.write(f"{parcela_catastral_joinkey}\n")

    def upload_image_generate_url(path):
            """
            Uploads an image to Google Cloud Storage and returns the public URL.
            If the image is already uploaded, it simply returns the existing URL.

            Args:
                path (str): The local path of the image to be uploaded.

            Returns:
                str: The public URL of the uploaded image or an error message if something goes wrong.
            """
            try:
                if pd.isnull(path):
                    return ''
                else:
                    tipo = path.split('/')[1]
                    finca = path.split('/')[2]

                    # Construct the blob path
                    blob_path = f'fotos_fincas/{tipo}/{finca}'
                    blob = bucket.blob(blob_path)

                    # Check if the blob already exists in the bucket
                    if blob.exists():
                        base_url = "https://storage.googleapis.com/building_images_storage/fotos_fincas/"
                        nombre_codificado = quote(finca)
                        url_completa = base_url + tipo + '/' + nombre_codificado
                        return url_completa

                    # If the blob doesn't exist, upload the image
                    blob.content_type = 'image/jpeg'
                    with open(path, 'rb') as f:
                        blob.upload_from_file(f)

                    # Generate the public URL for the uploaded image
                    base_url = "https://storage.googleapis.com/building_images_storage/fotos_fincas/"
                    nombre_codificado = quote(finca)
                    url_completa = base_url + tipo + '/' + nombre_codificado
                    return url_completa

            except Exception as e:
                return f'Error: {str(e)}'

    
def main():
   # API keys and configuration
    BASE_URL = 'https://api.airtable.com/v0/'
    api_key = 'patxufVpMMsrxbVsx.50c4bdb9a1efc2cacffe86fefd6fc399f59643bb24a0e2215988faf3b0f1cfd8'
    base_name = 'appqbJijymmUlJ3uu'
    GOOGLE_MAPS_API_KEY = 'AIzaSyAlgZ92OFztxC-xAOJKwKsWCESY_xFtWXE'

    BUCKET_NAME = 'building_images_storage'
    user_agent = 'myGeolocator'

    # Initialize Airtable API with BASE_URL
    airtable_api = Api(api_key, api_url=BASE_URL)
    airtable_base = Base(api_key, base_name, api_url=BASE_URL)

    # Load models
    modelyolo = YOLO("D:\Github\Sophiq\tasks\Facade Classification\segment.pt")  # Replace with your YOLO model path
    cmodel = tf.keras.models.load_model("D:\Github\Sophiq\tasks\Facade Classification\fine_tuned_building_classification_model.h5")  # Replace with your classification model path

    # Initialize FincaPhotoDownloader class
    downloader = FincaPhotoDownloader(
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
        api_key=api_key,
        base_name=base_name,
        bucket_name=BUCKET_NAME,
        user_agent=user_agent,
        modelyolo=modelyolo,
        cmodel=cmodel
    )

    try:
        # Step 1: Check if retraining is due
        if downloader.is_training_due():
            print("Monthly retraining is due. Starting the process...")

            # Step 2: Download and process new manually tagged images
            print("Downloading new manually tagged images...")
            downloader.download_new_manually_tagged_images()

            print("Updating dataset with new images...")

            # Step 3: Process and organize images
            print("Organizing and processing images...")
            downloader.process_images()
            downloader.organize_dataset(
                base_dir="./fotos_fincas/",
                output_dir="./processed_dataset/"
            )

            # Step 3.1: Apply segmentation to organized images
            print("Applying segmentation to organized images...")
            downloader.apply_segmentation_to_directory(
                directory="./processed_dataset",
                model=downloader.modelyolo,
                target_label="main_building"
            )

            # Step 4: Fine-tune the classification model
            print("Fine-tuning the classification model...")
            train_dir = "./processed_dataset/train"
            val_dir = "./processed_dataset/validation"
            downloader.fine_tune_model_monthly(train_dir, val_dir)

            # Step 5: Update training date in configuration
            downloader.update_training_date()
            print("Monthly retraining completed successfully!")

        else:
            print("Retraining is not due yet. Skipping retraining.")

        # Step 6: Apply model and generate predictions
        print("Generating predictions for unclassified fincas...")
        df_fincas_toSplit = downloader.download_df_fincas_nameproperly()
        dict_fotos_fincas = downloader.run_dict_fotos_fincas(df_fincas_toSplit)
        df_fincas_valoradas = downloader.generate_df_streetmaps_catastro_paths(dict_fotos_fincas)

        # Apply classification model to predict building type
        df_fincas_valoradas = downloader.apply_model_generate_df_classification_confidence(
            df_fincas_valoradas,
            downloader.classification_model,
            downloader.model,
            img_height=224,
            img_width=224,
            class_names=['noclasica', 'clasica']
        )

        # Step 7: Save predictions
        print("Saving predictions to file...")
        df_fincas_valoradas.to_csv("fincas_valoradas_with_predictions.csv", index=False)
        print("Pipeline completed successfully!")

    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")
if __name__ == "__main__":
    main()