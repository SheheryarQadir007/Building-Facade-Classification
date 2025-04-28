import logging
from pyairtable import Api
from ultralytics import YOLO
from Preprocessing import Preprocessing
from training import ImageProcessor
from utils import Utilities
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants (Replace with actual values)
# API keys and configuration
BASE_URL = 'https://api.airtable.com/v0/'
AIRTABLE_API_KEY = 'patxufVpMMsrxbVsx.50c4bdb9a1efc2cacffe86fefd6fc399f59643bb24a0e2215988faf3b0f1cfd8'
BASE_NAME = 'appqbJijymmUlJ3uu'
GOOGLE_MAPS_API_KEY = 'AIzaSyAlgZ92OFztxC-xAOJKwKsWCESY_xFtWXE'

BUCKET_NAME = 'building_images_storage'
USER_AGENT = 'myGeolocator'

def main():
    try:
        logging.info("Initializing components...")

        # Load models
        yolo_model = YOLO("segment.pt")

        try:
            classification_model = tf.keras.models.load_model("fine_tuned_builing_classification_model.h5")
        except:
            classification_model = tf.keras.models.load_model("building_classification_model_finetuned.h5")

        # Initialize utility class for common operations
        utils = Utilities(GOOGLE_MAPS_API_KEY)

        # Initialize data preprocessing class
        preprocessing = Preprocessing(
            AIRTABLE_API_KEY, BASE_NAME, GOOGLE_MAPS_API_KEY, BUCKET_NAME, USER_AGENT
        )

        # Step 1: Download finca data and process images
        logging.info("Fetching finca data and processing images...")
        df_fincas = preprocessing.update_validated_address_AI()
        print('df_fincas',len(df_fincas))

        df_fincas_to_split = preprocessing.download_df_fincas_nameproperly()
        print('df_fincas_to_split',len(df_fincas_to_split))
        preprocessing.download_photos_save_in_folders(df_fincas_to_split)

        # Initialize image processing and downloading
        finca_downloader = Preprocessing(
            GOOGLE_MAPS_API_KEY, AIRTABLE_API_KEY, BASE_NAME, BUCKET_NAME, USER_AGENT, yolo_model, classification_model
        )

        # Step 3: Organize and preprocess dataset
        logging.info("Organizing and preprocessing dataset...")
        img_processor = ImageProcessor(GOOGLE_MAPS_API_KEY, BUCKET_NAME)
        base_dir = "fotos_fincas"
        output_dir = "organized_dataset"
        img_processor.organize_dataset(base_dir, output_dir)

        # Step 4: Train model if needed
        if img_processor.is_training_due():
            logging.info("Training model...")
            train_dir = f"{output_dir}/train"
            val_dir = f"{output_dir}/validation"
            # Apply segmentation before model training
            logging.info("Applying segmentation to training and validation data...")
            img_processor.apply_segmentation_to_directory(train_dir, yolo_model, target_label="main_building")
            img_processor.apply_segmentation_to_directory(val_dir, yolo_model, target_label="main_building")
            img_processor.train_model(train_dir, val_dir, epochs=10, batch_size=32)
            img_processor.update_training_date()

        classification_model_new = tf.keras.models.load_model("fine_tuned_builing_classification_model.h5")

        finca_downloader = Preprocessing(
            GOOGLE_MAPS_API_KEY, AIRTABLE_API_KEY, BASE_NAME, BUCKET_NAME, USER_AGENT, yolo_model, classification_model_new
        )
        # Step 5: Apply segmentation and classification
        logging.info("Applying segmentation and classification...")
        df_fincas_valoradas = finca_downloader.run_dict_fotos_fincas(df_fincas_to_split)
        df_fincas_valoradas = img_processor.apply_model_generate_df_classification_confidence(
            df_fincas_valoradas, classification_model, yolo_model, 224, 224, ['noclasica', 'clasica']
        )

        # Step 6: Save predictions and upload images
        logging.info("Saving predictions and uploading images...")
        df_fincas_valoradas.to_csv("fincas_valoradas_with_predictions.csv", index=False)
        df_fincas_valoradas['GCP_URL'] = df_fincas_valoradas['streetmaps_path'].apply(img_processor.upload_image_generate_url)

        logging.info("Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
