import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

import os
from datetime import datetime

LOG_FILE =f"{datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)
logs_path = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(filename=logs_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO) 

