import os
import requests
import time
import pandas as pd
from google.cloud import storage, vision
from io import BytesIO
from urllib.parse import quote
import logging
from pyairtable import Api
from geopy.geocoders import Nominatim
import googlemaps
import re
from utils import Utitlities
class ImageDownloader:


    def __init__(self, google_maps_api_key, base_name, bucket_name, api_key, user_agent):
        self.google_maps_api_key = google_maps_api_key
        self.api = Api(api_key)
        self.base_name = base_name
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.geolocator = Nominatim(user_agent=user_agent)
        self.gmaps = googlemaps.Client(key=google_maps_api_key)
        self.utils = Utitlities(google_maps_api_key)




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
    def download_street_maps_photo(self, df, tipo, location, folder):
        """Downloads images from Google Street View API and uploads them to GCP bucket."""
        pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
        
        location = df.loc[df['parcela_catastral_joinkey'] == location, 'Address Validated AI'].values[0]
        par_catastral = df.loc[df['parcela_catastral_joinkey'] == location, 'Parcela Catastral'].values[0]

        pic_params = {
            "key": self.google_maps_api_key,
            "location": location,
            "size": "640x640",
            "pitch": "30",
            "source": "outdoor",
        }

        file_name = f"{par_catastral}.jpg"
        blob = self.client.bucket(self.bucket_name).blob(f'fotos_fincas/{tipo}/{file_name}')

        if blob.exists():
            logging.info(f"Image {par_catastral} already exists. Skipping download.")
            return

        try:
            pic_response = requests.get(pic_base, params=pic_params)
            pic_response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Error fetching image: {e}")
            time.sleep(120)
            return

        self._save_image(folder, tipo, file_name, pic_response.content)
        self._upload_to_gcp(os.path.join(folder, tipo, file_name), file_name)

    def _save_image(self, folder, tipo, filename, image_content):
        """Save the downloaded image locally."""
        os.makedirs(os.path.join(folder, tipo), exist_ok=True)
        file_path = os.path.join(folder, tipo, filename)
        with open(file_path, "wb") as file:
            file.write(image_content)
        logging.info(f"Image saved locally at {file_path}")

    def _upload_to_gcp(self, local_path, filename):
        """Upload the image to Google Cloud Storage."""
        blob = self.client.bucket(self.bucket_name).blob(f'fotos_fincas/{filename}')
        blob.upload_from_filename(local_path)
        logging.info(f"Uploaded {filename} to GCP storage.")


    def update_validated_addresses(self, df, city='Madrid'):
        """Update and validate addresses in the dataset."""
        df['Address Validated AI'] = df['Finca'].apply(lambda x: self.utils.validate_address(x, city))
        df['Codigo Postal'] = df['Address Validated AI'].apply(lambda x: self.utils.extract_postal_code(x))
        return df

