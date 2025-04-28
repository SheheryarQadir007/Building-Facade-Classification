import os
import json
import logging
import time
import shutil
import numpy as np
import pandas as pd
import cv2
import datetime
import requests
import tensorflow as tf
from pathlib import Path
from urllib.parse import quote
from google.cloud import storage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import albumentations as A
from tensorflow.keras.applications import MobileNetV2
from IPython.display import clear_output

# storage_client = storage.Client()
# bucket = storage_client.bucket(BUCKET_NAME)
class ImageProcessor:
    def __init__(self, google_maps_api_key, bucket_name, config_file="training_config.json", processed_log="processed_photos.log"):
        self.google_maps_api_key = google_maps_api_key
        self.bucket_name = bucket_name
        self.config_file = config_file
        self.processed_log = processed_log
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def segment_image(self, image_path, model, target_label="main_building"):
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(source=image_path, save=False, imgsz=640, iou=0.5)

        best_confidence = 0
        best_box = None
        for result in results:
            for box, confidence, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label_name = model.names[int(cls)]
                if label_name == target_label and confidence > best_confidence:
                    best_confidence = confidence
                    best_box = box.cpu().numpy().astype(int)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            x1, x2 = max(0, x1), min(image_rgb.shape[1], x2)
            y1, y2 = max(0, y1), min(image_rgb.shape[0], y2)
            return image_rgb[y1:y2, x1:x2]
        return image_rgb
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
                    segmented_image = self.segment_image(row.catastro_path, segmentation_model, target_label)
                    classification, confidence = self.predict_image(segmented_image, img_height, img_width, classification_model, class_names)
                    catastro_clasification_list.append(classification)
                    catastro_confidence_list.append(confidence)
                else:
                    catastro_clasification_list.append(np.NaN)
                    catastro_confidence_list.append(np.NaN)

                # Process Streetmaps image
                if row.streetmaps_path != '':
                    segmented_image = self.segment_image(row.streetmaps_path, segmentation_model, target_label)
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

  

    def upload_image_generate_url(self, path):
        if pd.isnull(path):
            return ''
        tipo = path.split('/')[1]
        finca = path.split('/')[2]
        blob_path = f'fotos_fincas/{tipo}/{finca}'
        blob = self.bucket.blob(blob_path)
        
        if not blob.exists():
            with open(path, 'rb') as f:
                blob.upload_from_file(f)
        return f"https://storage.googleapis.com/{self.bucket_name}/fotos_fincas/{tipo}/{quote(finca)}"

    def organize_dataset(self, base_dir, output_dir):
        clasica_dir = os.path.join(base_dir, "clasica")
        noclasica_dir = os.path.join(base_dir, "noclasica")
        
        if not os.path.exists(clasica_dir) or not os.path.exists(noclasica_dir):
            raise FileNotFoundError("Clasica or noclasica directory not found.")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        clasica_images = [os.path.join(clasica_dir, f) for f in os.listdir(clasica_dir) if f.endswith(('.jpg', '.png'))]
        noclasica_images = [os.path.join(noclasica_dir, f) for f in os.listdir(noclasica_dir) if f.endswith(('.jpg', '.png'))]

        train_clasica, temp_clasica = train_test_split(clasica_images, test_size=0.3, random_state=42)
        val_clasica, test_clasica = train_test_split(temp_clasica, test_size=0.5, random_state=42)

        train_noclasica, temp_noclasica = train_test_split(noclasica_images, test_size=0.3, random_state=42)
        val_noclasica, test_noclasica = train_test_split(temp_noclasica, test_size=0.5, random_state=42)

        subsets = ["train", "validation", "test"]
        categories = ["clasica", "noclasica"]

        for subset in subsets:
            for category in categories:
                os.makedirs(os.path.join(output_dir, subset, category), exist_ok=True)

        def copy_images(image_list, target_dir):
            for img in image_list:
                shutil.copy(img, target_dir)

        copy_images(train_clasica, os.path.join(output_dir, "train", "clasica"))
        copy_images(val_clasica, os.path.join(output_dir, "validation", "clasica"))
        copy_images(test_clasica, os.path.join(output_dir, "test", "clasica"))
        copy_images(train_noclasica, os.path.join(output_dir, "train", "noclasica"))
        copy_images(val_noclasica, os.path.join(output_dir, "validation", "noclasica"))
        copy_images(test_noclasica, os.path.join(output_dir, "test", "noclasica"))


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

    def train_model(self, train_dir, val_dir, epochs=50, batch_size=32, img_size=(224, 224)):
        """Train and evaluate the model with additional features like callbacks and logging."""

        print(f"Training the model for {epochs} epochs...")

        # Initialize image data generators for training and validation
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

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
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze the base model initially

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Define callbacks for model training
        lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Train the model with callbacks
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[lr_reduction, tensorboard_callback]
        )

        # Save the fine-tuned model
        model.save("building_classification_model_finetuned.h5")
        print("Model fine-tuned and saved successfully.")

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
