import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ...existing code...
import logging  
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
from src.exception import CustomException  
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:  
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')          

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  

    def initiate_data_ingestion(self):
        """
        Reads the dataset, splits into train/test, and saves the files.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            input_path = os.path.abspath('notebook/stud.csv')
            if not os.path.exists(input_path):
                logging.error(f"Input file not found: {input_path}")
                raise FileNotFoundError(f"Input file not found: {input_path}")

            df = pd.read_csv(input_path)
            if df.empty:
                logging.error("The input CSV file is empty.")
                raise ValueError("The input CSV file is empty.")

            logging.info('Read the dataset as dataframe')

            output_dir = os.path.dirname(self.ingestion_config.train_data_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Created directory: {output_dir}")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data is saved")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Performed train-test split")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data are saved")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error(f"Error occurred in data ingestion: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()