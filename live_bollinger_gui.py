import time
import numpy as np
import matplotlib.pyplot as plt

# Enable interactive mode for immediate plot updates
plt.ion()

class MultiBollingerChart:
    """
    A class that manages multiple Bollinger-like lines on a single Matplotlib figure.
    Each line is identified by a string name. You call push_new_data() with a dictionary:
        { "LineNameA": (valueA, stdevA),
          "LineNameB": (valueB, stdevB),
          ... }
    Each line is plotted with (value, upper band, lower band) sharing the same color,
    but the bands use partial transparency for a lighter shade.

    The chart only displays data from the last `t_window` seconds on the x-axis.
    """

    def __init__(self, t_window: float = 10.0, num_std: float = 2.0):
        """
        :param t_window: How many seconds of data to keep visible on the x-axis.
        :param num_std: Multiplier for standard deviations (upper/lower bands).
        """
        self.t_window = t_window
        self.num_std = num_std

        # Dictionary to store line data for each line_name:
        #   line_data[line_name] = {
        #       "times":    [],  # list of float timestamps
        #       "values":   [],  # list of float
        #       "stdevs":   [],  # list of float
        #       "line_val":   <matplotlib.lines.Line2D>,
        #       "line_upper": <matplotlib.lines.Line2D>,
        #       "line_lower": <matplotlib.lines.Line2D>
        #   }
        self.line_data = {}

        # Create figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.t_window)
        self.ax.set_xlabel("Time (seconds, recent window)")
        self.ax.set_ylabel("Value (+/- std dev)")

    def push_new_data(self, data_dict: dict[str, tuple[float, float]]):
        """
        Call this to append new data for one or more lines, by name.
        :param data_dict: A dictionary like:
            {
                "LineA": (valueA, stdA),
                "LineB": (valueB, stdB),
                ...
            }
        Each lineName in data_dict is updated with the provided (value, stdev).
        If a lineName is new, we'll create new line objects automatically.
        Then we immediately refresh the figure.
        """
        now = time.time()

        for line_name, (val, std) in data_dict.items():
            # If this line doesn't exist yet, create it
            if line_name not in self.line_data:
                self._create_line(line_name)

            # Append the new data to this line's lists
            self.line_data[line_name]["times"].append(now)
            self.line_data[line_name]["values"].append(val)
            self.line_data[line_name]["stdevs"].append(std)

        # Update the plot right away
        self.update_plot()

    def _create_line(self, line_name: str):
        """
        Internal helper to create a new line (value, upper, lower)
        for a given line_name with a unique color from Matplotlib.
        """
        # Create the main (value) line, letting Matplotlib pick a color
        line_val, = self.ax.plot([], [], label=f"{line_name} (value)")

        # Get that color so we can reuse it for the bands
        base_color = line_val.get_color()

        # Create upper/lower lines with the same color but more transparent
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

        # Update the legend to include these new lines
        self.ax.legend()

    def update_plot(self):
        """
        For each line, remove data older than (now - t_window),
        recalculate (value, upper, lower), and redraw the figure.
        """
        now = time.time()

        for line_name, data in self.line_data.items():
            # Filter data to keep only points within the last t_window
            cutoff = now - self.t_window

            times_array = np.array(data["times"])
            values_array = np.array(data["values"])
            stdevs_array = np.array(data["stdevs"])

            valid_mask = (times_array >= cutoff)
            times_filtered = times_array[valid_mask]
            vals_filtered = values_array[valid_mask]
            stds_filtered = stdevs_array[valid_mask]

            # Replace the original arrays with the filtered ones
            data["times"] = list(times_filtered)
            data["values"] = list(vals_filtered)
            data["stdevs"] = list(stds_filtered)

            if len(times_filtered) == 0:
                # No data left in the time window, clear the lines
                data["line_val"].set_data([], [])
                data["line_upper"].set_data([], [])
                data["line_lower"].set_data([], [])
                continue

            # Convert times to "relative" for the x-axis, so it starts at 0
            first_timestamp = times_filtered[0]
            x_values = times_filtered - first_timestamp

            # Bollinger-like lines
            upper = vals_filtered + (self.num_std * stds_filtered)
            lower = vals_filtered - (self.num_std * stds_filtered)

            # Update each line
            data["line_val"].set_data(x_values, vals_filtered)
            data["line_upper"].set_data(x_values, upper)
            data["line_lower"].set_data(x_values, lower)

        # Ensure x-axis always shows [0..t_window]
        self.ax.set_xlim(0, self.t_window)

        # Auto-scale the y-axis
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)

        # Redraw immediately
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    chart = MultiBollingerChart(t_window=10, num_std=2.0)

    # Let's simulate random updates for 3 lines:
    # "LineA", "LineB", and "LineC"
    line_names = ["LineA", "LineB", "LineC"]

    for i in range(30):
        data_dict = {}
        for ln in line_names:
            # e.g., offset the lines slightly to differentiate them
            offset = {"LineA": 0, "LineB": 15, "LineC": 30}[ln]
            val = 100 + offset + random.gauss(0, 1)
            std = random.uniform(0.5, 2.0)
            data_dict[ln] = (val, std)

        # Send the dictionary to the chart
        chart.push_new_data(data_dict)

        time.sleep(0.5)

    input("Press Enter to exit...")
