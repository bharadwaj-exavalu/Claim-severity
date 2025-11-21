import boto3

def upload_file_to_s3(file_name, bucket_name, object_name=None):
    """Uploads a file to an S3 bucket."""
    s3_client = boto3.client("s3")

    if object_name is None:
        object_name = file_name  # Use the same file name in S3

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"✅ File '{file_name}' uploaded successfully to '{bucket_name}/{object_name}'")
    except Exception as e:
        print(f"❌ Error uploading file: {e}")

if __name__ == "__main__":
    file_name = "synthetic_claim_data.csv"  # Replace with your dataset file path
    bucket_name = "claim-severity-trainnew"  # Replace with your bucket name
    upload_file_to_s3(file_name, bucket_name)
