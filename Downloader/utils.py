import os
import requests
import time
import pandas as pd
import numpy as np
from google.cloud import storage, vision
from io import BytesIO
from urllib.parse import quote
import logging
from pyairtable import Api
from geopy.geocoders import Nominatim
import googlemaps
import re




class Utilities:
    def __init__(self, google_maps_api_key):
        self.gmaps = googlemaps.Client(key=google_maps_api_key)


    def extract_postal_code(self, text):
        """Extract postal code from text."""
        try:
            numbers = re.findall(r'\d+', text)
            return str(max(int(num) for num in numbers if int(num) > 1000))
        except ValueError:
            return None
        


    def validate_address(self, address, city):
        """Validate address using Google Maps API."""
        try:
            response = self.gmaps.geocode(address + ', ' + city)
            if response:
                return response[0]['formatted_address']
            else:
                return 'Error'
        except Exception as e:
            logging.error(f"Error validating address {address}: {e}")
            return 'Error'


    def save_list_fast(self, my_list, file_path):
        with open(file_path, 'w') as f:
            f.write('\n'.join(my_list))

    
    
    def load_list_with_numpy(self, file_name):
        return np.loadtxt(f"{file_name}.txt", dtype=str).tolist()