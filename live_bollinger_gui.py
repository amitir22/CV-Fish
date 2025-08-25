"""Interactive plotting utilities for visualising metric trends.

The :class:`MultiPairBollingerChart` class provides a simple live plot
that displays metric values and their Bollinger bands for multiple frame
pairs.  A dropdown widget allows the user to switch between pairs.
"""

import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Dropdown


plt.ion()  # Interactive mode to update plots without blocking


class MultiPairBollingerChart:
    """Display Bollinger metrics for multiple frame pairs with a dropdown selector."""

    def __init__(self, pair_labels, t_window: float = 10.0, num_std: float = 2.0):
        """Initialise the chart.

        Parameters
        ----------
        pair_labels:
            Iterable of labels such as ``"1-2"`` identifying the frame
            pairs available for plotting.
        t_window:
            Length of the rolling time window to display.
        num_std:
            Number of standard deviations used to draw the Bollinger
            bands.
        """
        self.t_window = t_window
        self.num_std = num_std
        self.pair_labels = pair_labels
        self.current_pair = pair_labels[0]

        # Data storage
        self.line_data = {}
        self.pair_map = defaultdict(list)
        self.frames = {p: None for p in pair_labels}
        self.flows = {p: None for p in pair_labels}

        # Create figure and axes
        self.fig, (self.ax_boll, self.ax_img) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax_boll.set_xlim(0, self.t_window)
        self.ax_boll.set_xlabel("Time (seconds, recent window)")
        self.ax_boll.set_ylabel("Value (+/- std dev)")

        self.im = None
        self.quiver = None
        self.ax_img.set_title("Latest Frame + Flow")
        self.ax_img.axis("off")

        # Dropdown to choose pair
        ax_dropdown = self.fig.add_axes([0.25, 0.95, 0.5, 0.04])
        self.dropdown = Dropdown(ax_dropdown, "Pair", pair_labels, value=self.current_pair)
        self.dropdown.on_changed(self._on_pair_change)

    def _on_pair_change(self, label):
        """Callback fired when the user selects a different frame pair."""
        self.current_pair = label
        self._set_line_visibility()
        if self.frames[label] is not None:
            self._update_image(self.frames[label])
        if self.flows[label] is not None:
            self._update_flow_quiver(self.flows[label], step=60)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def push_new_data(
        self,
        data_dict: dict[str, tuple[float, float]] = None,
        frame: np.ndarray = None,
        flow: np.ndarray = None,
        pair_name: str = None,
    ):
        """Add a new metric sample and optionally update the image/flow display."""
        now = time.time()
        if pair_name is None:
            pair_name = self.current_pair

        if data_dict is not None:
            for line_name, (val, std) in data_dict.items():
                key = f"{pair_name}_{line_name}"
                if key not in self.line_data:
                    self._create_line(key)
                    self.pair_map[pair_name].append(key)
                ld = self.line_data[key]
                ld["times"].append(now)
                ld["values"].append(val)
                ld["stdevs"].append(std)

        if frame is not None:
            self.frames[pair_name] = frame
            if pair_name == self.current_pair:
                self._update_image(frame)

        if flow is not None:
            self.flows[pair_name] = flow
            if pair_name == self.current_pair:
                self._update_flow_quiver(flow, step=60)

        self._update_bollinger_plot()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _create_line(self, key: str):
        """Initialise plot lines to track a new metric."""
        line_val, = self.ax_boll.plot([], [], label=f"{key} (value)")
        base_color = line_val.get_color()
        line_up, = self.ax_boll.plot([], [], label=f"{key} (upper)", color=base_color, alpha=0.35)
        line_down, = self.ax_boll.plot([], [], label=f"{key} (lower)", color=base_color, alpha=0.35)
        self.line_data[key] = {
            "times": [],
            "values": [],
            "stdevs": [],
            "line_val": line_val,
            "line_upper": line_up,
            "line_lower": line_down,
        }
        self._set_line_visibility()

    def _set_line_visibility(self):
        """Show only the metrics related to the currently selected pair."""
        for pair, keys in self.pair_map.items():
            visible = pair == self.current_pair
            for key in keys:
                self.line_data[key]["line_val"].set_visible(visible)
                self.line_data[key]["line_upper"].set_visible(visible)
                self.line_data[key]["line_lower"].set_visible(visible)
        self.ax_boll.legend()

    def _update_bollinger_plot(self):
        """Redraw the Bollinger lines using the most recent data window."""
        now = time.time()
        cutoff = now - self.t_window
        for data in self.line_data.values():
            times_array = np.array(data["times"])
            vals_array = np.array(data["values"])
            stds_array = np.array(data["stdevs"])
            valid_mask = times_array >= cutoff
            times_filtered = times_array[valid_mask]
            vals_filtered = vals_array[valid_mask]
            stds_filtered = stds_array[valid_mask]
            data["times"] = list(times_filtered)
            data["values"] = list(vals_filtered)
            data["stdevs"] = list(stds_filtered)
            if len(times_filtered) == 0:
                data["line_val"].set_data([], [])
                data["line_upper"].set_data([], [])
                data["line_lower"].set_data([], [])
                continue
            x_values = times_filtered - cutoff
            upper = vals_filtered + self.num_std * stds_filtered
            lower = vals_filtered - self.num_std * stds_filtered
            data["line_val"].set_data(x_values, vals_filtered)
            data["line_upper"].set_data(x_values, upper)
            data["line_lower"].set_data(x_values, lower)
        self.ax_boll.set_xlim(0, self.t_window)
        self.ax_boll.relim()
        self.ax_boll.autoscale_view(scalex=False, scaley=True)

    def _update_image(self, frame: np.ndarray):
        """Display the latest frame in the side panel."""
        if self.im is None:
            self.im = self.ax_img.imshow(frame)
        else:
            self.im.set_data(frame)
        self.ax_img.set_title("Latest Frame + Flow")
        self.ax_img.axis("off")

    def _update_flow_quiver(self, flow: np.ndarray, step: int = 15):
        """Overlay a sparse quiver plot of optical flow vectors on the frame."""
        if flow.ndim != 3 or flow.shape[2] != 2:
            print("Flow must be shape (H, W, 2). Skipping quiver.")
            return
        if self.quiver is not None:
            self.quiver.remove()
            self.quiver = None
        H, W, _ = flow.shape
        y_indices, x_indices = np.mgrid[0:H:step, 0:W:step]
        X = x_indices.flatten()
        Y = y_indices.flatten()
        U = flow[Y, X, 0]
        V = flow[Y, X, 1]
        self.quiver = self.ax_img.quiver(
            X,
            Y,
            U,
            V,
            color='red',
            units='xy',
            angles='xy',
            scale_units='xy',
            scale=0.05,
            pivot='tail',
            width=0.005,
            headwidth=10,
            headlength=12,
            headaxislength=8,
            linewidths=0.7,
            edgecolors='yellow',
            alpha=1.0,
            zorder=2,
        )
        self.ax_img.set_xlim(0, W)
        self.ax_img.set_ylim(H, 0)


if __name__ == "__main__":
    chart = MultiPairBollingerChart(["1-2", "1-3", "1-4", "1-5"], t_window=10, num_std=2.0)
    import random
    for i in range(5):
        for pair in ["1-2", "1-3", "1-4", "1-5"]:
            val = random.random()
            std = random.random()
            frame = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
            flow = np.random.randn(100, 150, 2).astype(np.float32)
            chart.push_new_data({"Farneback_magnitude_mean": (val, std)}, frame=frame, flow=flow, pair_name=pair)
        time.sleep(0.5)
    input("Press Enter to exit...")

