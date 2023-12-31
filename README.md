# Cōnstellātiō
This is the repository used by the chem-ensamble project. The name "cōnstellātiō" means "set of stars" in Latin.

Constellatio is developed by Alán Aspuru-Guzik's group at the University of Toronto.

## Add a new model 

Here, you can add a model to be trained by the orion network. To do so, follow these steps:

1. Under `models/`, create a new folder `my_folder`. Here, you can add all the models you would like to deploy. If you already have a folder, skip to step 3.

2. Create an `__init__.py` file for your module. Then, add it to `models/info_models.py` with
    ```python
    # Import your model
    import constellatio.models.my_folder as my_folder

    ...
    # Add your module to the list of modules
    model_modules = [..., my_folder]
    ```

3. Create a `my_model.py` file under `models/my_folder`. Here, you can define your `nn.Module` (or any subclass of it, such as `LightningModule`). 
    ```python
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Define your model here

        def forward(self, x):
            # Define your forward pass here
            return x
    ```

4. Add the model defined in `my_model.py` to `models/my_folder/__init__.py`
    ```python
    from .my_model import MyModel
    ```

## Add a new dataset

Here, you can add a new dataset to be used by the orion network. To do so, follow these steps:

1. Create a new file under `datasets/` with the name of your dataset.

2. [WIP]


### Authors:
- [Luis Mantilla](https://github.com/BestQuark)
- [Marko Huang](https://linkedin.com/in/markohuang)