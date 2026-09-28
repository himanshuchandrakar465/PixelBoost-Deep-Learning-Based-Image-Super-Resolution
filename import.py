import kagglehub
from dotenv import load_dotenv

load_dotenv(r"C:\Users\ASUS\Desktop\A_GOOD_PROJECT\.env")

# Download latest version
path = kagglehub.dataset_download("joe1995/div2k-dataset")

print("Path to dataset files:", path)
