ulimit -n 10240

export AWS_ACCESS_KEY_ID='obAZIu0I0q5gI32bVT1D'
export AWS_SECRET_ACCESS_KEY='nuzmiz-gAjtah-tinse0'
export MLFLOW_S3_ENDPOINT_URL=http://192.168.2.30:9020
export MLFLOW_S3_IGNORE_TLS=true 
export MLFLOW_TRACKING_URI=http://192.168.2.30:5000 

# Train using the Wikitext config
# accelerate launch src/train.py --config_file configs/config_server.yaml
accelerate launch expertlm/train.py --config_file configs/config_aisrv02.yaml