import constellatio.models.marko as marko
import constellatio.models.luis as luis
import constellatio.models.other as other
# Add your model module here

from torch import nn
import inspect

# Add your model module here
model_modules = [marko, luis, other]

all_models = {}

for module in model_modules:
    for name, obj in vars(module).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            all_models[name] = obj
