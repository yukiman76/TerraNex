ulimit -n 10240
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,2,3
# Train using the Wikitext config
# First, install the package in development mode if not already installed
pip install -e . 

# Then run the training script
python -m expertlm.train --config_file configs/config_test.yaml

# or
# accelerate launch src/train.py --config_file configs/config_test.yaml