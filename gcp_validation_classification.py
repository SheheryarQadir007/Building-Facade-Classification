import json
from sklearn.model_selection import train_test_split
# Disable oneDNN optimizations before any TensorFlow imports.


import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
import os
import requests
import time
import pandas as pd
import re
import airtable_download_upload
from pyairtable import Api, Base, Table
from unidecode import unidecode
import datetime
import os
import shutil
from urllib.parse import quote

from geopy.geocoders import GoogleV3
from google.cloud import vision
from geopy.geocoders import Nominatim
import googlemaps
from collections import defaultdict
import logging
from ultralytics import YOLO

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_io as tfio
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Global configuration parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 1
TRAINING_INTERVAL_DAYS = 7
 

class GCPValidationClassification:
    def __init__(self, model_path=None):

        load_dotenv()
        self.GOOGLE_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.GOOGLE_CREDENTIALS
        self.BUCKET_NAME = "building_images_storage"
        self.CONFIG_FILE_NAME = os.getenv("CONFIG_FILE_NAME")
        self.GOOGLE_MAPS_API_KEY = 'AIzaSyAlgZ92OFztxC-xAOJKwKsWCESY_xFtWXE'
        self.geolocator = Nominatim(user_agent="myGeolocator")
        self.gmaps = googlemaps.Client(key=self.GOOGLE_MAPS_API_KEY)
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.BUCKET_NAME)
        self.BASE_URL = 'https://api.airtable.com/v0/'
        self.api_key = 'patxufVpMMsrxbVsx.50c4bdb9a1efc2cacffe86fefd6fc399f59643bb24a0e2215988faf3b0f1cfd8'
        self.base_name = 'appqbJijymmUlJ3uu'
        self.api = Api(self.api_key)
        self.model_path = model_path
        self.training_metadata_blob = "training_metadata.json"

        # Load the model immediately
        try:
            self.model = tf.keras.models.load_model(model_path)
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise




    def create_dic(self):
        # Create a dictionary to store folders and their files
        folder_files = defaultdict(list)

        # List all blobs in the bucket
        blobs = self.bucket.list_blobs()

        # Organize files by folder
        for blob in blobs:
            # Skip if blob is a folder placeholder
            if blob.name.endswith('/'):
                continue

            # Get folder path by removing filename
            folder = os.path.dirname(blob.name)
            if folder == '':
                folder = 'root'

            # Add filename to appropriate folder list
            filename = os.path.basename(blob.name)
            folder_files[folder].append(filename)

        # Convert defaultdict to regular dict
        folder_files = dict(folder_files)
        return folder_files

    def ascii_filename(self, file_name):
        # Convert non-ASCII characters to their closest ASCII counterparts
        name, extension = os.path.splitext(file_name)
        ascii_name = unidecode(name)
        # Replace spaces with underscores and remove non-alphanumeric characters
        ascii_name = ''.join(e for e in ascii_name if e.isalnum() or e in ('-', '_')).rstrip()
        return f"{ascii_name}{extension}"

    def validate_address(self, address, city):
        try:
            location = self.gmaps.addressvalidation([address],
                                                    regionCode='ES',
                                                    locality=city)['result']['address']['formattedAddress']
            return location

        except:
            return 'Error'

    def extract_postal_code(text):
        try:
            numbers = re.findall(r'\d+', text)
            large_numbers = [int(num) for num in numbers if int(num) > 1000]
            return str(int(large_numbers[0]))
        except:
            return None

    def download_df_fincas_nameproperly(self):
        table = self.api.table(self.base_name, 'Fincas')
        columns_to_include = [
            ['FINCA', 'Año construcción', 'Tipo Finca', 'Parcela Catastral', 'parcela_catastral_joinkey',
             'Codigo Postal', 'Address Validated AI']]
        formula = "{parcela_catastral_joinkey}"
        table = table.all(formula=formula, fields=columns_to_include)
        df_fincas_toSplit = pd.DataFrame(table)
        lista_records = list(df_fincas_toSplit['id'])
        df_fincas_toSplit = pd.DataFrame(list(df_fincas_toSplit['fields']))
        df_fincas_toSplit['record_id'] = lista_records
        df_fincas_toSplit['Codigo Postal'] = df_fincas_toSplit['Codigo Postal'].apply(
            lambda x: int(x) if pd.notna(x) else None).fillna('')
        df_fincas_toSplit['Finca_Proper_google'] = df_fincas_toSplit['FINCA'].str.title() + ', Madrid'
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['parcela_catastral_joinkey'].notna()]
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['Address Validated AI'].notna()]
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['Finca_Proper_google'].notna()]
        df_fincas_toSplit['ascii_filename'] = df_fincas_toSplit['Address Validated AI'].map(self.ascii_filename)

        dict_tipo_fincas = {'Representativa +5%': 'Clasica',
                            'Clásica +0%': 'Clasica',
                            'Moderna -10%': 'Moderna',
                            'Moderna-Clásica -5%': 'Moderna',
                            'Asintónica -20%': 'Moderna'}

        df_fincas_toSplit['Tipo Finca'] = df_fincas_toSplit['Tipo Finca'].replace(dict_tipo_fincas)
        return df_fincas_toSplit

    def finca_get_street_maps_photo_download(self, df, tipo, location):
        """
        Downloads the Street View and cadastral images for a given location and uploads
        them directly to the GCP bucket under the appropriate folder based on 'tipo'.

        The images are uploaded to:
            fotos_fincas/{tipo}/{parcela_catastral_joinkey}.jpg          (Street View image)
            fotos_fincas/{tipo}/{parcela_catastral_joinkey}_catastro.jpg   (Cadastral image)

        Args:
            df (pd.DataFrame): DataFrame containing the address and cadastral information.
            tipo (str): Classification type (should be one of "clasica", "noclasica", or "no_valorada").
            location (str): The key for the location (typically the parcela_catastral_joinkey).
            folder (str): (Unused in this version) Previously the local folder; now the bucket destination is fixed.
        """

        # Determine values from the DataFrame.
        parcela_catastral_joinkey = location
        if tipo == 'no_valorada':
            par_catastral = df[df['parcela_catastral_joinkey'] == location]['Parcela Catastral'].unique()[0]
            location = df[df['parcela_catastral_joinkey'] == location]['Address Validated AI'].unique()[0]
        else:
            par_catastral = df[df['parcela_catastral_joinkey'] == location]['Parcela Catastral'].unique()[0]
            location = df[df['parcela_catastral_joinkey'] == location]['Address Validated AI'].unique()[0]
        print(f"Got the {tipo} for location: {location}")

        # ---------------------
        # Download Street View Image
        # ---------------------
        meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
        pic_base = 'https://maps.googleapis.com/maps/api/streetview?'

        meta_params = {'key': self.GOOGLE_MAPS_API_KEY, 'location': location}
        pic_params = {
            'key': self.GOOGLE_MAPS_API_KEY,
            'location': location,
            'size': "640x640",
            'pitch': '30',
            'source': 'outdoor'
        }
        try:
            print(f"[DEBUG] Attempting to fetch Street View image for location: {location}")
            pic_response = requests.get(pic_base, params=pic_params)
            print(f"[DEBUG] Fetched Street View image. Status Code: {pic_response.status_code}")
        except Exception as e:
            print(f"[DEBUG] Error fetching Street View image: {e}. Retrying in 120 seconds.")
            time.sleep(120)
            try:
                print(f"[DEBUG] Retrying Street View image fetch for location: {location}")
                pic_response = requests.get(pic_base, params=pic_params)
                print(f"[DEBUG] Retry successful. Status Code: {pic_response.status_code}")
            except Exception as retry_e:
                print(f"[DEBUG] Retry failed: {retry_e}. Skipping this location.")
                return

        # Upload the Street View image directly to GCS.
        try:
            # Construct the blob path:
            # e.g. "fotos_fincas/clasica/<parcela_catastral_joinkey>.jpg"
            street_view_blob_path = f"fotos_fincas/{tipo}/{parcela_catastral_joinkey}.jpg"
            self.bucket.blob(street_view_blob_path).upload_from_string(
                pic_response.content, content_type='image/jpeg'
            )
            print(f"[DEBUG] Uploaded Street View image to {street_view_blob_path}")
        except Exception as e:
            print(f"[DEBUG] Error uploading Street View image to GCS: {e}. Skipping this location.")
        pic_response.close()

        # ---------------------
        # Download and Upload Cadastral Image
        # ---------------------
        if type(par_catastral) is not float:
            cadastral_url = (
                    'http://ovc.catastro.meh.es/OVCServWeb/OVCWcfLibres/OVCFotoFachada.svc/'
                    'RecuperarFotoFachadaGet?ReferenciaCatastral=' + par_catastral
            )
            print(f"[DEBUG] Constructed cadastral image URL: {cadastral_url}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/58.0.3029.110 Safari/537.3'
            }
            print(f"[DEBUG] Using headers for cadastral image request: {headers}")
            print("[DEBUG] Sending GET request for cadastral image.")
            try:
                cadastral_response = requests.get(cadastral_url, headers=headers)
                print(f"[DEBUG] Received cadastral image. Status Code: {cadastral_response.status_code}")
            except Exception as e:
                print(f"[DEBUG] Exception during GET for cadastral image: {e}")
                raise

            # Create a Vision API image object.
            image = vision.Image(content=cadastral_response.content)
            print("[DEBUG] Created Vision API image object for cadastral image.")

            client = vision.ImageAnnotatorClient()
            try:
                print("[DEBUG] Initiating label detection on cadastral image.")
                vision_response = client.label_detection(image=image)
                print("[DEBUG] Label detection completed.")
                labels = vision_response.label_annotations
                print(f"[DEBUG] Detected {len(labels)} labels.")
                label_saved = False
                for idx, label in enumerate(labels, start=1):
                    print(f"[DEBUG] Processing label {idx}: '{label.description}' with score {label.score}")
                    if label.description == 'Building' and label.score >= 0.7:
                        print(f"[DEBUG] 'Building' label confirmed with score {label.score}. Saving cadastral image.")
                        # Construct blob path for cadastral image.
                        # e.g. "fotos_fincas/clasica/<parcela_catastral_joinkey>_catastro.jpg"
                        cadastral_blob_path = f"fotos_fincas/{tipo}/{parcela_catastral_joinkey}_catastro.jpg"
                        self.bucket.blob(cadastral_blob_path).upload_from_string(
                            cadastral_response.content, content_type='image/jpeg'
                        )
                        print(f"[DEBUG] Uploaded cadastral image to {cadastral_blob_path}")
                        label_saved = True
                        break
                if not label_saved:
                    # If no 'Building' label detected with sufficient confidence, still upload the image.
                    cadastral_blob_path = f"fotos_fincas/{tipo}/{parcela_catastral_joinkey}catastro.jpg"
                    self.bucket.blob(cadastral_blob_path).upload_from_string(
                        cadastral_response.content, content_type='image/jpeg'
                    )
                    print(f"[DEBUG] Uploaded cadastral image without label confirmation to {cadastral_blob_path}")
            except Exception as e:
                print(
                    f"[DEBUG] Exception during label detection: {e}. Uploading cadastral image without label confirmation.")
                cadastral_blob_path = f"fotos_fincas/{tipo}/{parcela_catastral_joinkey}_catastro.jpg"
                self.bucket.blob(cadastral_blob_path).upload_from_string(
                    cadastral_response.content, content_type='image/jpeg'
                )
                print(f"[DEBUG] Uploaded cadastral image to {cadastral_blob_path}")

    def check_image_existence(self, folder_files, parcela_catastral_joinkey):
        print(f"[DEBUG] Checking existence of images for parcela_catastral_joinkey: {parcela_catastral_joinkey}")

        for folder, files in folder_files.items():
            print(f"[DEBUG] Inspecting folder: {folder}")

            expected_images = [
                f"{parcela_catastral_joinkey}.jpg",
                f"{parcela_catastral_joinkey}catastro.jpg"
            ]
            print(f"[DEBUG] Expected images: {expected_images}")

            for img in expected_images:
                print(f"[DEBUG] Checking for image: {img} in folder: {folder}")
                if img in files:
                    print(f"[DEBUG] Image {img} found in folder {folder}")
                    return True, folder
                else:
                    print(f"[DEBUG] Image {img} not found in folder {folder}")

        print(f"[DEBUG] Images for {parcela_catastral_joinkey} are missing in all folders")
        return False, None

    def check_images_for_sampled_data(self, folder_files, sampled_data):
        print("[DEBUG] Starting check_images_for_sampled_data")
        not_found_records = []

        for joinkey in sampled_data['parcela_catastral_joinkey']:
            print(f"[DEBUG] Checking images for joinkey: {joinkey}")
            result, folder = self.check_image_existence(folder_files, joinkey)
            if not result:
                print(f"[DEBUG] Images for joinkey {joinkey} not found")
                not_found_records.append(joinkey)
            else:
                print(f"[DEBUG] Images for joinkey {joinkey} found in folder {folder}")
            print(f"[DEBUG] Checked {joinkey}: {'Found in ' + folder if result else 'Not Found'}")

        # Create a DataFrame for not found records
        print("[DEBUG] Creating DataFrame for not found records")
        not_found_df = sampled_data[sampled_data['parcela_catastral_joinkey'].isin(not_found_records)]
        print(f"[DEBUG] Total missing records: {len(not_found_records)}")
        return not_found_df

    def download_missing_images(self, missing_images_df, df):
        print("[DEBUG] Starting download_missing_images")

        for _, row in missing_images_df.iterrows():
            joinkey = row['parcela_catastral_joinkey']
            tipo_finca = str(row['Tipo Finca']).strip().lower()
            print(f"[DEBUG] Processing joinkey: {joinkey}, Tipo Finca: {tipo_finca}")

            if tipo_finca in ['', 'nan', 'none']:
                tipo = 'no_valorada'
                print(f"[DEBUG] Tipo Finca is empty or invalid. Set tipo to 'no_valorada'")
            elif tipo_finca == 'clasica':
                tipo = 'clasica'
                print(f"[DEBUG] Tipo Finca is 'clasica'. Set tipo to 'clasica'")
            else:
                tipo = 'noclasica'
                print(f"[DEBUG] Tipo Finca is '{tipo_finca}'. Set tipo to 'noclasica'")

            try:
                print(f"[DEBUG] Calling finca_get_street_maps_photo_download for joinkey: {joinkey}")
                self.finca_get_street_maps_photo_download(df, tipo, joinkey)
                print(f"[DEBUG] Successfully called finca_get_street_maps_photo_download for joinkey: {joinkey}")
            except FileNotFoundError as e:
                print(f"Error saving image: {e}")
                print(f"[DEBUG] FileNotFoundError encountered for joinkey: {joinkey}. Continuing to next record.")
                continue
            except Exception as e:
                print(f"[DEBUG] Unexpected error while downloading image for {joinkey}: {e}")
                continue


    def segment_image_array(self, image: np.ndarray, model, target_label: str = "main_building", threshold: float = 0.80) -> np.ndarray:
        """
        Runs YOLO prediction on the input image and draws the best bounding box (with label and confidence)
        for the target label ("main_building") if its confidence is at least the threshold.
        Otherwise, crops the image from the center by removing 15% margins.
        This version works entirely in-memory.
        """
        # Convert image to RGB for prediction.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            results = model.predict(source=image_rgb, save=False, imgsz=640, iou=0.5)
        except Exception as e:
            logging.error(f"Error during YOLO prediction: {e}")
            return image_rgb

        best_confidence = 0
        best_box = None
        any_detection = False

        # Loop through all predictions and track the best detection for the target label.
        for result in results:
            for box, confidence, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label_name = model.names[int(cls)]
                if label_name == target_label:
                    any_detection = True
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_box = box.cpu().numpy().astype(int)

        if any_detection and best_box is not None and best_confidence >= threshold:
            x1, y1, x2, y2 = best_box
            height, width, _ = image_rgb.shape
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                logging.warning("Invalid bounding box; will crop center region instead.")
            else:
                annotated = image_rgb.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{target_label} {best_confidence:.2f}"
                cv2.putText(annotated, text, (x1, max(y1-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                logging.info(f"Detection: {best_box} with confidence {best_confidence:.2f}")
                return annotated

        # If no valid detection is found, crop the center of the image by removing 15% margins.
        logging.info("No valid detection with confidence >= 0.80; cropping center region.")
        height, width, _ = image_rgb.shape
        x1 = int(0.15 * width)
        y1 = int(0.15 * height)
        x2 = int(0.85 * width)
        y2 = int(0.85 * height)
        cropped_center = image_rgb[y1:y2, x1:x2]
        return cropped_center

    def segment_and_upload_directories(self, input_dirs: list, model) -> None:
        """
        Processes images in the specified GCS directories, applies YOLO segmentation (using the logic above),
        and uploads the processed images (annotated or center-cropped) to GCS.

        The output structure will be:
          - Input:  gs://building_images_storage/fotos_fincas/clasica/
                   Output: gs://building_images_storage/segmented_fotos_fincas/clasica/
          - Input:  gs://building_images_storage/fotos_fincas/noclasica/
                   Output: gs://building_images_storage/segmented_fotos_fincas/noclasica/
          - Input:  gs://building_images_storage/fotos_fincas/no_valorada/
                   Output: gs://building_images_storage/segmented_fotos_fincas/no_valorada/

        Only 15 images per directory are processed.

        Args:
            input_dirs (list): List of GCS directory URIs containing image files.
            model (YOLO): Loaded YOLO model for segmentation.
        """
        allowed_extensions = (".jpg", ".jpeg", ".png")
        all_output_blobs = []  # Collect output blob URIs.

        for input_dir in input_dirs:
            if not input_dir.startswith("gs://"):
                logging.error(f"Invalid GCS URI: {input_dir}")
                continue

            # Parse the folder name (e.g., "clasica", "noclasica", "no_valorada").
            parts = input_dir.split('/')
            if len(parts) < 4:
                logging.error(f"Could not parse bucket and folder from URI: {input_dir}")
                continue
            folder_name = parts[-2]
            logging.info(f"Processing directory: {input_dir} (folder: {folder_name})")

            # Define the output prefix.
            output_prefix = f"segmented_fotos_fincas/{folder_name}/"

            # Remove the bucket prefix to get the blob prefix.
            input_prefix = input_dir.replace(f"gs://{self.bucket_name}/", "")
            blobs = self.bucket.list_blobs(prefix=input_prefix)
            count = 0

            for blob in blobs:
                if count >= 3:
                    break
                if not blob.name.lower().endswith(allowed_extensions):
                    continue

                logging.info(f"Processing blob: {blob.name}")
                try:
                    image_bytes = blob.download_as_bytes()
                except Exception as e:
                    logging.error(f"Error downloading blob {blob.name}: {e}")
                    continue

                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    logging.error(f"Failed to decode image from blob: {blob.name}")
                    continue

                # Process the image: annotate with detection (if available) or crop center.
                processed_image = self.segment_image_array(image, model, target_label="main_building", threshold=0.80)

                # Convert processed image from RGB to BGR for JPEG encoding.
                processed_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                success, buffer = cv2.imencode('.jpg', processed_bgr)
                if not success:
                    logging.error(f"Failed to encode processed image for blob: {blob.name}")
                    continue
                processed_bytes = buffer.tobytes()

                # Determine output file name.
                filename = blob.name.split('/')[-1]
                output_blob_name = f"{output_prefix}{filename}"
                try:
                    out_blob = self.bucket.blob(output_blob_name)
                    out_blob.upload_from_string(processed_bytes, content_type='image/jpeg')
                    full_output_uri = f"gs://{self.BUCKET_NAME}/{output_blob_name}"
                    logging.info(f"Uploaded processed image to: {full_output_uri}")
                    all_output_blobs.append(full_output_uri)
                except Exception as e:
                    logging.error(f"Error uploading processed image to {output_blob_name}: {e}")
                    continue

                count += 1

        # Print all output blob paths.
        print("\nProcessed images uploaded to the following GCS paths:")
        for output_uri in all_output_blobs:
            print(output_uri)

    def list_blobs_in_prefix(self, prefix: str, allowed_extensions=(".jpg", ".jpeg", ".png")):
        """
        Lists all blobs in the bucket with the given prefix, filtering by allowed image extensions.
        """
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        filtered = [blob for blob in blobs if blob.name.lower().endswith(allowed_extensions)]
        return filtered

    def copy_blobs(self, blob_list, output_prefix: str):
        """
        Copies each blob in blob_list to the destination under output_prefix within the same bucket.
        Returns a list of destination URIs.
        """
        output_uris = []
        for blob in blob_list:
            filename = blob.name.split('/')[-1]
            dest_blob_name = f"{output_prefix}{filename}"
            self.bucket.copy_blob(blob, self.bucket, dest_blob_name)
            output_uri = f"gs://{self.BUCKET_NAME}/{dest_blob_name}"
            output_uris.append(output_uri)
            logging.info(f"Copied {blob.name} to {dest_blob_name}")
        return output_uris

    def train_val_split_gcs(self):
        """
        Splits images from the segmented folders ("segmented_fotos_fincas/clasica/" and
        "segmented_fotos_fincas/noclasica/") into training and validation sets,
        and copies them into the following output structure:

            gs://building_images_storage/training_fotos_fincas/train/clasica/
            gs://building_images_storage/training_fotos_fincas/validation/clasica/
            gs://building_images_storage/training_fotos_fincas/train/noclasica/
            gs://building_images_storage/training_fotos_fincas/validation/noclasica/

        The split is done with 70% of images for training and 30% for validation.
        """
        # Define source prefixes.
        source_clasica = "segmented_fotos_fincas/clasica/"
        source_noclasica = "segmented_fotos_fincas/noclasica/"

        # List blobs from each category.
        clasica_blobs = self.list_blobs_in_prefix(source_clasica)
        noclasica_blobs = self.list_blobs_in_prefix(source_noclasica)
        logging.info(f"Found {len(clasica_blobs)} clasica images and {len(noclasica_blobs)} noclasica images.")

        # Split each category into train (70%) and validation (30%).
        train_clasica, val_clasica = train_test_split(clasica_blobs, test_size=0.3, random_state=42)
        train_noclasica, val_noclasica = train_test_split(noclasica_blobs, test_size=0.3, random_state=42)

        # Define output prefixes.
        output_prefixes = {
            "train_clasica": "training_fotos_fincas/train/clasica/",
            "val_clasica": "training_fotos_fincas/validation/clasica/",
            "train_noclasica": "training_fotos_fincas/train/noclasica/",
            "val_noclasica": "training_fotos_fincas/validation/noclasica/"
        }

        # Copy blobs to the respective output locations.
        out_train_clasica = self.copy_blobs(train_clasica, output_prefixes["train_clasica"])
        out_val_clasica = self.copy_blobs(val_clasica, output_prefixes["val_clasica"])
        out_train_noclasica = self.copy_blobs(train_noclasica, output_prefixes["train_noclasica"])
        out_val_noclasica = self.copy_blobs(val_noclasica, output_prefixes["val_noclasica"])

        # Print output URIs.
        print("\nTraining Clasica:")
        for uri in out_train_clasica:
            print(uri)
        print("\nValidation Clasica:")
        for uri in out_val_clasica:
            print(uri)
        print("\nTraining Noclasica:")
        for uri in out_train_noclasica:
            print(uri)
        print("\nValidation Noclasica:")
        for uri in out_val_noclasica:
            print(uri)

    def get_last_training_date(self):
        """Get last training date from GCS"""
        try:
            blob = self.bucket.blob(self.training_metadata_blob)
            if blob.exists():
                content = blob.download_as_string()
                data = json.loads(content)
                return datetime.fromisoformat(data['last_training'])
            return None
        except Exception as e:
            logging.warning(f"Error reading last training date: {e}")
            return None

    def should_train(self):
        """Check if enough time has passed since last training"""
        last_training = self.get_last_training_date()
        if last_training is None:
            logging.info("No previous training date found. Will proceed with training.")
            return True
            
        days_since_training = (datetime.now() - last_training).days
        if days_since_training < TRAINING_INTERVAL_DAYS:
            logging.info(f"Only {days_since_training} days since last training. "
                       f"Waiting for {TRAINING_INTERVAL_DAYS - days_since_training} more days.")
            return False
            
        logging.info(f"{days_since_training} days since last training. Will proceed with training.")
        return True

    def update_training_date(self):
        """Update the last training date in GCS"""
        try:
            metadata = {
                'last_training': datetime.now().isoformat(),
                'model_path': self.model_path
            }
            blob = self.bucket.blob(self.training_metadata_blob)
            blob.upload_from_string(json.dumps(metadata))
            logging.info("Updated last training date in GCS")
        except Exception as e:
            logging.error(f"Failed to update training date in GCS: {e}")

    def get_file_list_from_gcs(self, directory, allowed_extensions=(".jpg", ".jpeg", ".png")):
        prefix = directory.replace(f"gs://{self.BUCKET_NAME}/", "")
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        file_list = [f"gs://{self.BUCKET_NAME}/{blob.name}"
                     for blob in blobs if blob.name.lower().endswith(allowed_extensions)]
        return file_list

    def get_dataset(self, directory, img_size):
        file_list = self.get_file_list_from_gcs(directory)
        if not file_list:
            raise FileNotFoundError(f"Directory {directory} not found or is empty.")
        logging.info(f"Found {len(file_list)} files in {directory}.")

        classes = sorted(list({os.path.basename(os.path.dirname(path)) for path in file_list}))
        logging.info(f"Inferred classes: {classes}")

        labels = [classes.index(os.path.basename(os.path.dirname(path))) for path in file_list]
        dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

        def load_and_preprocess(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, img_size)
            image = image / 255.0
            label = tf.one_hot(label, depth=len(classes))
            return image, label

        return dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    def retrain(self, train_dir, val_dir):
        """Retrain the model if enough time has passed"""
        if not self.should_train():
            return None

        logging.info("Starting retraining process...")
        
        # Set up augmentations
        train_transforms = A.Compose([
            A.RandomResizedCrop(*IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.OneOf([A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5)], p=0.5),
            A.GridDropout(ratio=0.2, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize()
        ])

        val_transforms = A.Compose([
            A.Resize(*IMG_SIZE),
            A.Normalize()
        ])

        # Prepare datasets
        train_dataset = self.get_dataset(train_dir, IMG_SIZE)
        val_dataset = self.get_dataset(val_dir, IMG_SIZE)

        def augment(image, label, transforms):
            def _augment(img):
                img = img.numpy()
                augmented = preprocess_image(img, transforms)
                return augmented.astype(np.float32)
            augmented_image = tf.py_function(func=_augment, inp=[image], Tout=tf.float32)
            augmented_image.set_shape((*IMG_SIZE, 3))
            return augmented_image, label

        train_dataset = (train_dataset
                        .map(lambda x, y: augment(x, y, train_transforms))
                        .batch(BATCH_SIZE)
                        .prefetch(tf.data.AUTOTUNE))

        val_dataset = (val_dataset
                      .map(lambda x, y: augment(x, y, val_transforms))
                      .batch(BATCH_SIZE)
                      .prefetch(tf.data.AUTOTUNE))

        if hasattr(self.model.layers[0], 'trainable'):
            self.model.layers[0].trainable = True
            logging.info("Unfroze base model")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            TensorBoard(log_dir="./logs")
        ]

        history = self.model.fit(
            train_dataset,
            epochs=FINE_TUNE_EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks
        )

        # Save model to GCS
        model_blob = self.bucket.blob(self.model_path)
        self.model.save(self.model_path)
        model_blob.upload_from_filename(self.model_path)
        logging.info(f"Retrained model saved to gs://{self.BUCKET_NAME}/{self.model_path}")
        
        # Update the last training date
        self.update_training_date()

        return self.model_path, history 


    def run_dict_fotos_fincas(self, df_fincas_toSplit):
        # Define the GCS directories to search.
        directories = [
            "gs://building_images_storage/segmented_fotos_fincas/clasica/",
            "gs://building_images_storage/segmented_fotos_fincas/noclasica/",
            "gs://building_images_storage/segmented_fotos_fincas/no_valorada/"
        ]

        # Build a dictionary to hold the filenames available in each GCS directory.
        dir_files = {}
        for directory in directories:
            # Remove the bucket prefix to get the internal prefix.
            prefix = directory.replace(f"gs://{self.BUCKET_NAME}/", "")
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            # Collect just the filename (the last part of the blob name) for each blob.
            files_set = set(blob.name.split("/")[-1] for blob in blobs if not blob.name.endswith("/"))
            dir_files[directory] = files_set

        dict_fotos_fincas = {}

        # Loop through each unique property key.
        for parcela_catastral_joinkey in df_fincas_toSplit.parcela_catastral_joinkey.unique():
            # Get the corresponding record_id from the DataFrame.
            record_id = df_fincas_toSplit[df_fincas_toSplit['parcela_catastral_joinkey'] == parcela_catastral_joinkey]['record_id'].unique()[0]
            
            # List to store found file paths.
            list_files_fincas = []
            
            # Prepare expected filenames.
            expected_filenames = [
                parcela_catastral_joinkey + 'catastro.jpg',
                parcela_catastral_joinkey + '.jpg'
            ]
            
            # Loop through each directory and check if any expected filename exists.
            for directory in directories:
                for fname in expected_filenames:
                    if fname in dir_files[directory]:
                        # Append the full GCS URI for the file.
                        list_files_fincas.append(f"{directory}{fname}")
            
            dict_fotos_fincas[record_id] = list_files_fincas

        return dict_fotos_fincas



    def generate_df_streetmaps_catastro_paths(self, dict_fotos_fincas):
        df_fincas_valoradas  = pd.DataFrame()

        record_id_list = []
        paths_list = []
        catastro_list = []
        streetmaps_list = []
        i = 0

        for record_id, paths in dict_fotos_fincas.items():
            i += 1
            
            record_id_list.append(record_id)
            paths_list.append(paths)
                    
        df_fincas_valoradas['record_id'] = record_id_list
        df_fincas_valoradas['paths'] = paths_list

        df_fincas_valoradas[['catastro_path', 'streetmaps_path']] = df_fincas_valoradas['paths'].apply(self.separate_paths)
        return df_fincas_valoradas

    def separate_paths(self, paths):
        catastro_path = None
        other_path = None
        
        for path in paths:
            if path.endswith("catastro.jpeg") or path.endswith("catastro.jpg"):
                catastro_path = path
            else:
                other_path = path
                
        return pd.Series([catastro_path, other_path])



    def imread_unicode(self, filename, flags=cv2.IMREAD_COLOR):
        """
        Reads an image from a file using a Unicode-friendly approach.
        This function reads the file as a byte array and decodes it with OpenCV.
        
        Args:
            filename (str): The path to the image file.
            flags (int): Flags for cv2.imdecode (default is cv2.IMREAD_COLOR).
        
        Returns:
            np.ndarray or None: The loaded image, or None if loading fails.
        """
        try:
            if not os.path.exists(filename):
                logging.error(f"File does not exist: {filename}")
                return None
            # Read file content as bytes
            img_array = np.fromfile(filename, np.uint8)
            img = cv2.imdecode(img_array, flags)
            return img
        except Exception as e:
            logging.error(f"Error reading image with Unicode support: {e}")
            return None

    def predict_image_from_array(self, image_array, img_height, img_width, model, class_names):
        """
        Processes an image array (e.g., a segmented image), resizes it to the required dimensions,
        and applies the classification model. It returns the predicted class name and the associated confidence.

        Args:
            image_array (np.ndarray): The image array (segmented image).
            img_height (int): Target image height.
            img_width (int): Target image width.
            model (tf.keras.Model): The trained classification model.
            class_names (list): List of class names corresponding to model outputs.

        Returns:
            tuple: (predicted_class, confidence)
        """
        # Resize the image to match the model's input shape.
        resized_image = cv2.resize(image_array, (img_width, img_height))
        # Normalize pixel values to [0, 1] (adjust preprocessing as required by your model)
        image = resized_image.astype('float32') / 255.0
        # Expand dimensions to create a batch of 1
        image = np.expand_dims(image, axis=0)
        # Get model predictions
        predictions = model.predict(image)
        # Get the class index with the highest probability
        pred_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][pred_idx]
        predicted_class = class_names[pred_idx]
        return predicted_class, confidence

    def apply_model_generate_df_classification_confidence(self, df_fincas_valoradas, model, yolo_model, img_height, img_width, class_names):
        """
        For each row in the provided DataFrame, this function segments the images (for both the 
        catastro and streetmaps paths) using the YOLO segmentation model. Then it uses the classification 
        model on the segmented image to get a class prediction and a confidence score.
        
        The results are stored in new columns of the DataFrame.

        Args:
            df_fincas_valoradas (pd.DataFrame): DataFrame containing at least 'catastro_path' and 'streetmaps_path' columns.
            model (tf.keras.Model): The classification model.
            yolo_model (YOLO): The segmentation model (YOLO) for cropping images.
            img_height (int): Target image height for classification.
            img_width (int): Target image width for classification.
            class_names (list): List of class names corresponding to model outputs.

        Returns:
            pd.DataFrame: The updated DataFrame with new columns for classification and confidence scores.
        """
        catastro_classification_list = []
        catastro_confidence_list = []
        streetmaps_classification_list = []
        streetmaps_confidence_list = []

        # Iterate over each row in the DataFrame
        for _, row in df_fincas_valoradas.iterrows():
            # Process catastro image if path is available
            if row.catastro_path != '':
                if row.catastro_path is not None:
                    classification, confidence = self.predict_image_from_array(row.catastro_path, img_height, img_width, model, class_names)
                    catastro_classification_list.append(classification)
                    catastro_confidence_list.append(confidence)
                else:
                    catastro_classification_list.append(np.NaN)
                    catastro_confidence_list.append(np.NaN)
            else:
                catastro_classification_list.append(np.NaN)
                catastro_confidence_list.append(np.NaN)

            # Process streetmaps image if path is available
            if row.streetmaps_path != '':
                if row.streetmaps_path is not None:
                    classification, confidence = self.predict_image_from_array(row.streetmaps_path, img_height, img_width, model, class_names)
                    streetmaps_classification_list.append(classification)
                    streetmaps_confidence_list.append(confidence)
                else:
                    streetmaps_classification_list.append(np.NaN)
                    streetmaps_confidence_list.append(np.NaN)
            else:
                streetmaps_classification_list.append(np.NaN)
                streetmaps_confidence_list.append(np.NaN)

        # Add the prediction results to the DataFrame
        df_fincas_valoradas['catastro_classification'] = catastro_classification_list
        df_fincas_valoradas['catastro_confidence'] = catastro_confidence_list
        df_fincas_valoradas['streetmaps_classification'] = streetmaps_classification_list
        df_fincas_valoradas['streetmaps_confidence'] = streetmaps_confidence_list

        return df_fincas_valoradas


def main():

    #Read CSV File
    # data = pd.read_csv('updated_sampled_data.csv')

    #Initialize Class
    model_path = "fine_tuned_building_classification_model.h5"
    validator = GCPValidationClassification(model_path)

    #Step 0: Prepare Dataset
    df_fincas_toSplit = validator.download_df_fincas_nameproperly()
    # print(df_fincas_toSplit)

    #Step 1: Prepare GCP Dictionary
    folder_files = validator.create_dic()
    # print(folder_files)
    #Step 2: Check for the missing Images
    print("[DEBUG] Starting image existence check for sampled data")
    missing_images_df = validator.check_images_for_sampled_data(folder_files, df_fincas_toSplit)
    #Step 3: Download Missing Images
    validator.download_missing_images(missing_images_df, df_fincas_toSplit)

    #Step 4: Model Retraining
    #Step 4.1: Appy Segmentation to directories
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Update this path to point to your YOLO model file
    model_path = "segment.pt"
    try:
        yolo_model = YOLO(model_path)
        logging.info("YOLO model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        exit(1)

    # List of input directories to process. Update these paths as needed.
    input_directories = [
        "gs://building_images_storage/fotos_fincas/clasica/",
        "gs://building_images_storage/fotos_fincas/noclasica/",
        "gs://building_images_storage/fotos_fincas/no_valorada/"
    ]

    # # Process the directories and save segmented images in new directories
    validator.segment_and_upload_directories(input_directories, yolo_model)

    # # Step 4.2: Apply Train Test Split
    validator.train_val_split_gcs()

    # # Step 4.3: Apply Retraining
    train_dir = "gs://building_images_storage/training_fotos_fincas/train/"  # e.g., "data/train"
    val_dir = "gs://building_images_storage/training_fotos_fincas/validation/"      # e.g., "data/val"

    model_path, history = validator.retrainer.retrain(train_dir, val_dir)
    updated_model = tf.keras.models.load_model(model_path)
    if history is None:
        logging.info("Training skipped - will try again later")
    else:
        logging.info("Training completed successfully")
    #Step 5: Classification
    
    dict_fotos_fincas = validator.run_dict_fotos_fincas(df_fincas_toSplit)
    df_fincas_paths = validator.generate_df_streetmaps_catastro_paths(dict_fotos_fincas)
    img_height = 224
    img_width = 224
    class_names = ['noclasica', 'clasica'] 
    df_fincas_valoradas = validator.apply_model_generate_df_classification_confidence(df_fincas_paths, updated_model, yolo_model, img_height, img_width, class_names)

if __name__ == '__main__':
    main()
    