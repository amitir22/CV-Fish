import time
import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # interactive mode on

class MultiBollingerChart:
    """
    A class that manages multiple Bollinger-like lines on a single Matplotlib figure
    AND displays an optional image next to the chart.

    Each time you push_new_data(), you can supply:
      1) A dictionary of line_name -> (value, stdev) for the Bollinger chart
      2) An optional np.ndarray image frame to be displayed in the right subplot

    The chart only displays data from the last `t_window` seconds on the x-axis.
    """

    def __init__(self, t_window: float = 10.0, num_std: float = 2.0):
        """
        :param t_window: How many seconds of data to keep visible on the x-axis.
        :param num_std: Multiplier for standard deviations (upper/lower bands).
        """
        self.t_window = t_window
        self.num_std = num_std

        # Dictionary storing line data for each line_name
        self.line_data = {}

        # Create a figure with 2 subplots: left for Bollinger, right for image
        self.fig, (self.ax, self.ax_img) = plt.subplots(1, 2, figsize=(10, 5))

        # Set up the Bollinger axis
        self.ax.set_xlim(0, self.t_window)
        self.ax.set_xlabel("Time (seconds, recent window)")
        self.ax.set_ylabel("Value (+/- std dev)")

        # We'll keep track of an image handle on the right axis
        # Initialize it to None
        self.im = None
        self.ax_img.set_title("Latest Frame")
        self.ax_img.axis("off")  # Hide axis ticks for the image

    def push_new_data(
        self, 
        data_dict: dict[str, tuple[float, float]], 
        frame: np.ndarray | None = None
    ):
        """
        Call this to append new data for one or more Bollinger lines and optionally
        show a new image on the right subplot.

        :param data_dict: A dictionary like:
            {
                "LineA": (valueA, stdA),
                "LineB": (valueB, stdB),
                ...
            }
          Each lineName is updated with the provided (value, stdev).
          If a lineName is new, we'll create new line objects automatically.
        :param frame: An optional np.ndarray that will be displayed
                      in the right subplot (e.g., a grayscale or RGB image).
                      If None, we don't update the image.
        """
        # TODO: implement a bollinger clock graph aswell for the angular data

        now = time.time()

        # 1) Update Bollinger data
        for line_name, (val, std) in data_dict.items():
            if line_name not in self.line_data:
                self._create_line(line_name)

            self.line_data[line_name]["times"].append(now)
            self.line_data[line_name]["values"].append(val)
            self.line_data[line_name]["stdevs"].append(std)

        # 2) Update the Bollinger plot
        self.update_bollinger_plot()

        # 3) If we have a frame, update the image on the right
        if frame is not None:
            self.update_image(frame)

        # 4) Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _create_line(self, line_name: str):
        """
        Internal helper to create a new line (value, upper, lower)
        for a given line_name with a unique color from Matplotlib.
        """
        line_val, = self.ax.plot([], [], label=f"{line_name} (value)")
        base_color = line_val.get_color()

        line_up, = self.ax.plot([], [], label=f"{line_name} (upper)",
                                color=base_color, alpha=0.35)
        line_down, = self.ax.plot([], [], label=f"{line_name} (lower)",
                                  color=base_color, alpha=0.35)

        self.line_data[line_name] = {
            "times": [],
            "values": [],
            "stdevs": [],
            "line_val": line_val,
            "line_upper": line_up,
            "line_lower": line_down
        }

        self.ax.legend()

    def update_bollinger_plot(self):
        """
        Removes data older than now - t_window for each line,
        then updates line data for plotting.
        """
        now = time.time()

        for line_name, data in self.line_data.items():
            times_array = np.array(data["times"])
            vals_array = np.array(data["values"])
            stds_array = np.array(data["stdevs"])

            cutoff = now - self.t_window
            valid_mask = (times_array >= cutoff)

            times_filtered = times_array[valid_mask]
            vals_filtered = vals_array[valid_mask]
            stds_filtered = stds_array[valid_mask]

            # Overwrite with the filtered data
            data["times"] = list(times_filtered)
            data["values"] = list(vals_filtered)
            data["stdevs"] = list(stds_filtered)

            if len(times_filtered) == 0:
                # No data => clear lines
                data["line_val"].set_data([], [])
                data["line_upper"].set_data([], [])
                data["line_lower"].set_data([], [])
                continue

            # # Convert times to 0..t_window scale
            # first_timestamp = times_filtered[0]
            # x_values = times_filtered - first_timestamp

            # Instead of subtracting the earliest timestamp, 
            # align the oldest data to x=0 and newest to x~t_window:
            offset = now - self.t_window
            x_values = times_filtered - offset
        
            upper = vals_filtered + self.num_std * stds_filtered
            lower = vals_filtered - self.num_std * stds_filtered

            data["line_val"].set_data(x_values, vals_filtered)
            data["line_upper"].set_data(x_values, upper)
            data["line_lower"].set_data(x_values, lower)

        # Adjust x-axis and y-axis
        self.ax.set_xlim(0, self.t_window)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)

    def update_image(self, frame: np.ndarray):
        """
        Displays the given 'frame' (numpy array) in the right subplot.
        If it's the first time, we create an image handle; otherwise, we update it.
        """
        if self.im is None:
            # Create the image handle the first time
            # Let imshow infer color map if it's 2D or display as RGB if it's 3D
            self.im = self.ax_img.imshow(frame)
        else:
            # Update the existing image
            self.im.set_data(frame)

        # Adjust the axis if needed (especially for a first-time shape)
        self.ax_img.set_title("Latest Frame")
        self.ax_img.axis("off")  # turn off axis ticks
        # If you want dynamic color scaling, call 'set_clim' or re-imshow
        # e.g.: self.im.set_clim(vmin=frame.min(), vmax=frame.max())


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    chart = MultiBollingerChart(t_window=10, num_std=2.0)

    line_names = ["LineA", "LineB"]
    for i in range(30):
        data_dict = {}
        for ln in line_names:
            offset = {"LineA": 0, "LineB": 15}[ln]
            val = 100 + offset + random.gauss(0, 1)
            std = random.uniform(0.5, 2.0)
            data_dict[ln] = (val, std)

        # Create a random image frame (e.g., 100x100 grayscale)
        # You could also have an RGB image (100x100x3).
        frame = np.random.randint(0, 255, (100, 100)).astype(np.uint8)

        # Push the new Bollinger data + the new image frame
        chart.push_new_data(data_dict, frame=frame)

        time.sleep(0.5)

    input("Press Enter to exit...")
