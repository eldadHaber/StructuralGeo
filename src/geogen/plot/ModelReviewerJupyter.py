"""A Jupyter notebook based single-model review tool/object for GeoWord models.

Requirements:
A Jupyter notebook environment with the PyVista and ipywidgets packages installed.

Parameters:
- generate_model_func: function that returns a new model instance
- base_dir: directory where models will be saved
- show_history: whether to show the history of the model in the output widget
- single_view: whether to show the model from a single view or a set of 6 views

Button Functions:
- Save Model: Save the current model and refresh the review.
- Discard Model: Discard the current model and refresh the review.
- Adjust Model Height: Adjust the height of the current model using auto-normalization.
- Exit Review: Exit the review and close the plotter.
"""

import math

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pyvista as pv
from IPython.display import clear_output, display

import geogen.plot as geovis
from geogen.filemanagement import FileManager
from geogen.model.util import rotate


class ModelReviewer:
    def __init__(
        self,
        generate_model_func,
        base_dir="../saved_models",
        show_history=True,
        single_view=False,
    ):
        """
        Initialize the ModelReviewer with specific model generation and plotting functions.

        Parameters
        - generate_model_func: function that returns a new model instance
        - base_dir: directory where models will be saved
        """
        self.generate_model = generate_model_func
        self.base_dir = base_dir
        self.output = widgets.Output()
        self.single_view = single_view
        self.plotter = pv.Plotter(off_screen=True)  # Create an off-screen plotter
        self.plotter.add_axes(line_width=5)
        self.fm = FileManager(base_dir=self.base_dir)
        self.show_history = show_history

    def init_buttons(self):
        """Initialize the control buttons and setup event handlers."""
        self.save_button = widgets.Button(description="Save Model")
        self.discard_button = widgets.Button(description="Discard Model")
        self.renormalize_button = widgets.Button(description="Adjust Model Height")
        self.exit_button = widgets.Button(description="Exit Review")
        self.save_button.on_click(self.save_action)
        self.discard_button.on_click(self.discard_action)
        self.renormalize_button.on_click(self.renormalize_action)
        self.exit_button.on_click(self.exit_action)
        self.button_box = widgets.HBox(
            [
                self.save_button,
                self.discard_button,
                self.renormalize_button,
                self.exit_button,
            ]
        )
        display(self.button_box)  # Display buttons below the plot

    def plot_model(self):
        with self.output:
            clear_output(wait=True)  # Clear the output before starting to plot the new model
            self.remove_all_actors(self.plotter)
            geovis.volview(self.current_model, plotter=self.plotter, show_bounds=True)

            if self.single_view:
                # INFO: For some outrageous reason the PyVista plotter needs to be shown twice or it will
                # lag by one render and show the previous mesh. This is sadly the workaround for now.
                self.plotter.show(window_size=[600, 400], jupyter_backend="static")
                clear_output(wait=True)
                self.plotter.show(window_size=[600, 400], jupyter_backend="static")
            else:
                screenshots = []
                for i in range(4):
                    cp = self.plotter.camera.position
                    M = rotate([0, 0, 1], math.pi / 2)
                    new_cp = M @ cp
                    self.plotter.camera_position = new_cp
                    screenshot = self.plotter.screenshot(return_img=True)
                    screenshots.append(screenshot)

                for i in range(2):
                    cp = self.plotter.camera.position
                    M = rotate([0, 1, 0], math.pi / 4)
                    new_cp = M @ cp
                    self.plotter.camera_position = new_cp
                    screenshot = self.plotter.screenshot(return_img=True)
                    screenshots.append(screenshot)

                # Display the screenshots in a 2x3 grid
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                axs = axs.flatten()
                for ax, screenshot in zip(axs, screenshots):
                    ax.imshow(screenshot)
                    ax.axis("off")
                plt.show()

                # Reset the camera position to the initial position
                self.plotter.view_isometric()

    def remove_all_actors(self, plotter):
        """Remove all actors from the plotter."""
        plotter.clear_actors()
        plotter.clear_plane_widgets()

    def refresh_model(self):
        new_model = self.generate_model()  # Generate new model
        self.current_model = new_model  # Update current model
        self.plot_model()  # Plot the new model
        if self.show_history:
            with self.output:
                print(self.current_model.get_history_string())

    def save_action(self, b):
        """Save the current model and refresh the review."""
        self.fm.save_geo_model(self.current_model, self.base_dir)  # Save the current model
        with self.output:
            print(f"Model saved to {self.base_dir}")
            self.refresh_model()  # Refresh to get a new model displayed

    def discard_action(self, b):
        """Discard the current model and refresh the review."""
        print("Model discarded.")
        self.refresh_model()  # Refresh to get a new model displayed

    def renormalize_action(self, b):
        """Adjust the height of the current model and refresh the review."""
        self.current_model.renormalize_height(auto=True)  # Adjust the height of the model
        self.plot_model()
        print("Model height adjusted.")

    def exit_action(self, b):
        """Exit the review, close the plotter, and remove all UI components."""
        self.plotter.close()  # Close the plotter to free up resources
        self.output.close()  # Close the output widget to remove it from the display
        self.button_box.close()  # Close the button box widget to remove it from the display
        print("Review exited.")

    def start_review(self):
        self.init_buttons()  # Initialize buttons
        self.refresh_model()  # Start the review by displaying the first model
        display(self.output)  # Display the output widget which will contain everything
