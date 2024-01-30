import sys
from constellatio.models.info_models import all_models

__all__ = ["all_models"] + list(all_models.keys())

for name, model in all_models.items():
    setattr(sys.modules[__name__], name, model)

if "model" in dir(sys.modules[__name__]):
    delattr(sys.modules[__name__], "model")

if "name" in dir(sys.modules[__name__]):
    delattr(sys.modules[__name__], "name")

# This dictionary contains
# value: unique_id of the model
# key: (model, port number for server, url_path for client, classification of the model)
