import json

def load_config(config_file):
    """Carica il file di configurazione JSON."""
    with open(config_file, 'r') as f:
        return json.load(f)