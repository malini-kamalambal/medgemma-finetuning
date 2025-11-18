from google.cloud import storage

# Configuration
project_id = "gcp-yard" 
bucket_name = "medgemma-tuning-demo" # Must be globally unique
source_file_name = "data.jsonl"
destination_blob_name = "training_data/data.jsonl"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=project_id)
    
    # Create the bucket if it doesn't exist
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists.")
    except:
        print(f"Creating bucket {bucket_name}...")
        bucket = storage_client.create_bucket(bucket_name, location="us-central1")

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}")

# Run the upload
upload_blob(bucket_name, source_file_name, destination_blob_name)