import boto3

def create_s3_bucket(bucket_name, region="us-east-1"):
    """Creates an S3 bucket in the specified AWS region."""
    s3_client = boto3.client("s3")

    try:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        print(f"✅ Bucket '{bucket_name}' created successfully in {region}")
    except Exception as e:
        print(f"❌ Error creating bucket: {e}")

if __name__ == "__main__":
    bucket_name = "claim-severity-trainnew"  # Change this to a unique name
    region = "us-east-1"  # Change if needed
    create_s3_bucket(bucket_name, region)
