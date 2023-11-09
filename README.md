# Cōnstellātiō
This is the repository used by the chem-ensamble project. The name "cōnstellātiō" means "set of stars" in Latin.

Constellatio is developed by Alán Aspuru-Guzik's group at the University of Toronto.

## Add a new model to the network

Here, you can add a model to be trained by the orion network. To do so, follow these steps:

1. Under `models/`, create a new folder with your name. Here, you can add all the models you would like to deploy. If there is already a folder with the name of your model, skip to step 3.

2. Create an `__init__.py` file for your module. Then, add it to `models/info_models.py` by including `import constellatio.models.my_name as my_name` and adding `my_name` to the `model_modules` list. 

3. Create a `my_model.py` file under `models/my_name`. Here, you can define your `nn.Module` (or any subclass of it, such as `LightningModule`). 

4. Add the model defined in `my_model.py` to the `__init__.py` file of `models/my_name`.

## Add a new dataset

Here, you can add a new dataset to be used by the orion network. To do so, follow these steps:

1. Create a new file under `datasets/` with the name of your dataset.

2. [WIP]


### Authors:
- [Luis Mantilla](https://github.com/BestQuark)
- [Marko Huang](https://linkedin.com/in/markohuang)