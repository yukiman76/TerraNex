ulimit -n 10240
# Train using the Wikitext config
python src/train.py --config_file configs/config_test.yaml
