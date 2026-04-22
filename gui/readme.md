# Pickled GeoModel GUI

This is a GUI in development for loading and viewing GeoModels that have been saved as a .pkl file. Saving of geomodels can be done using the `filemanagement` module in the `StructuralGeo` package. Models can be saved in their "lean" form by default where the mesh and data arrays are not saved, but can be regenerated from the model history.

## Loading Models
Use the `File-> Select Models Folder` to select a folder with .pkl files to view. In the file tree on the left, simply select the file that you want to view and it will automatically compile and display in the plotter window.

## Toolbar
The bottom has a context dependent toolbar depending on the plotter view.
- **Volume View** the height can be renormalized, model saved, a view of 2d slices, and the slices can also be saved to file.
- **N-Slice View** allows for looking at a cross section of slices along a particular axis. The number of slices can be adjusted.
- **Transformation View** allows for viewing the sequence of mesh transformations that created the model, along with the intermediary deposition states.
- **Categorical Grid View** (experimental) allows for viewing individual categories. Right now the category input is not validated and choosing a category that doesn't exist will crash the program.

## Shortcuts
- `Ctrl + S` will save the model in its current state. The model save path is still handled in source code but can be edited in the toolbar file
- `H` will renormalize the height of the model.
