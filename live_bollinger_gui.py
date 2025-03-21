import time
import numpy as np
import matplotlib.pyplot as plt
import cv_fish_configuration as conf

plt.ion()  # Interactive mode to update plots without blocking

class MultiBollingerChart:
    """
    A class that manages multiple Bollinger-like lines on the left subplot
    and displays a frame + optical flow overlay on the right subplot.

    Each call to push_new_data() can provide:
      - A dictionary of line_name -> (value, std) for Bollinger lines
      - A frame (np.ndarray) to display in the right subplot
      - A flow (np.ndarray, shape=(H, W, 2)) to overlay via quiver
    """

    def __init__(self, t_window: float = 10.0, num_std: float = 2.0):
        """
        :param t_window: How many seconds of data to keep visible on the x-axis.
        :param num_std: Multiplier for standard deviations (upper/lower bands).
        """
        self.t_window = t_window
        self.num_std = num_std

        # For Bollinger data:
        self.line_data = {}

        # Create a figure with 2 subplots: left for Bollinger, right for image/flow
        self.fig, (self.ax_boll, self.ax_img) = plt.subplots(1, 2, figsize=(10, 5))

        # Set up Bollinger axis
        self.ax_boll.set_xlim(0, self.t_window)
        self.ax_boll.set_xlabel("Time (seconds, recent window)")
        self.ax_boll.set_ylabel("Value (+/- std dev)")

        # For the image + flow overlay
        self.im = None              # Will be a handle to the displayed image
        self.quiver = None          # Will be a handle to the quiver plot
        self.ax_img.set_title("Latest Frame + Flow")
        self.ax_img.axis("off")     # Hide axis ticks on the image subplot

    def push_new_data(
        self, 
        data_dict: dict[str, tuple[float, float]] = None,
        frame: np.ndarray = None,
        flow: np.ndarray = None
    ):
        """
        Adds new Bollinger data (line_name -> (value, std)), plus optionally
        updates the displayed frame and flow in the right subplot.

        :param data_dict: e.g. {
            "LineA": (valA, stdA),
            "LineB": (valB, stdB)
        }
        :param frame: An optional image (2D or 3D np.ndarray) to display
        :param flow: An optional dense flow array of shape (H, W, 2)
                     from e.g. cv2.calcOpticalFlowFarneback
        """
        now = time.time()

        # 1) Update Bollinger lines
        if data_dict is not None:
            for line_name, (val, std) in data_dict.items():
                if line_name not in self.line_data:
                    self._create_line(line_name)
                self.line_data[line_name]["times"].append(now)
                self.line_data[line_name]["values"].append(val)
                self.line_data[line_name]["stdevs"].append(std)

        self._update_bollinger_plot()

        # 2) Update the frame on the right
        if frame is not None:
            self._update_image(frame)

        # 3) Overlay flow quiver on the same right subplot
        if flow is not None:
            # Overlays the flow vectors on top of the current image
            # (If no frame was provided, it still draws on the right axis)
            self._update_flow_quiver(flow, step=60)

        # 4) Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _create_line(self, line_name: str):
        """
        Internal helper to create a new line (value, upper, lower)
        for a given line_name with a unique color from Matplotlib.
        """
        line_val, = self.ax_boll.plot([], [], label=f"{line_name} (value)")
        base_color = line_val.get_color()

        line_up, = self.ax_boll.plot([], [], label=f"{line_name} (upper)",
                                     color=base_color, alpha=0.35)
        line_down, = self.ax_boll.plot([], [], label=f"{line_name} (lower)",
                                       color=base_color, alpha=0.35)

        self.line_data[line_name] = {
            "times": [],
            "values": [],
            "stdevs": [],
            "line_val": line_val,
            "line_upper": line_up,
            "line_lower": line_down
        }

        self.ax_boll.legend()

    def _update_bollinger_plot(self):
        """
        Removes data older than now - t_window for each line,
        then updates line data for plotting.
        """
        now = time.time()
        cutoff = now - self.t_window

        for line_name, data in self.line_data.items():
            times_array = np.array(data["times"])
            vals_array = np.array(data["values"])
            stds_array = np.array(data["stdevs"])

            valid_mask = (times_array >= cutoff)
            times_filtered = times_array[valid_mask]
            vals_filtered = vals_array[valid_mask]
            stds_filtered = stds_array[valid_mask]

            data["times"] = list(times_filtered)
            data["values"] = list(vals_filtered)
            data["stdevs"] = list(stds_filtered)

            if len(times_filtered) == 0:
                # Clear lines if no data in the window
                data["line_val"].set_data([], [])
                data["line_upper"].set_data([], [])
                data["line_lower"].set_data([], [])
                continue

            # SHIFT X to 0..t_window so the oldest point is x=0, newest is ~ x=t_window
            x_values = times_filtered - cutoff

            upper = vals_filtered + self.num_std * stds_filtered
            lower = vals_filtered - self.num_std * stds_filtered

            data["line_val"].set_data(x_values, vals_filtered)
            data["line_upper"].set_data(x_values, upper)
            data["line_lower"].set_data(x_values, lower)

        # Keep x-axis [0..t_window]
        self.ax_boll.set_xlim(0, self.t_window)

        # Autoscale y
        self.ax_boll.relim()
        self.ax_boll.autoscale_view(scalex=False, scaley=True)

    def _update_image(self, frame: np.ndarray):
        """
        Displays the given 'frame' (numpy array) in the right subplot.
        If it's the first time, we create an image handle; otherwise, we update it.
        """
        if self.im is None:
            self.im = self.ax_img.imshow(frame)
        else:
            self.im.set_data(frame)

        # Adjust axis
        self.ax_img.set_title("Latest Frame + Flow")
        self.ax_img.axis("off")

    def _update_flow_quiver(self, flow: np.ndarray, step: int = 15):
        """
        Overlays a quiver (arrow) plot of the flow on top of the right subplot.
        :param flow: np.ndarray of shape (H, W, 2), each pixel has (flow_x, flow_y).
        :param step: sampling distance for quiver (to avoid too many arrows)
                     e.g., step=15 means we sample every 15 pixels in x and y.
        """
        if flow.ndim != 3 or flow.shape[2] != 2:
            print("Flow must be shape (H, W, 2). Skipping quiver.")
            return

        # Remove old quiver (if any)
        if self.quiver is not None:
            self.quiver.remove()
            self.quiver = None

        H, W, _ = flow.shape

        # Generate a coarse grid of sample points
        # e.g. we take y=range(0,H,step), x=range(0,W,step)
        y_indices, x_indices = np.mgrid[0:H:step, 0:W:step]

        # Flatten them for quiver
        X = x_indices.flatten()
        Y = y_indices.flatten()

        # Flow vectors (U,V)
        U = flow[Y, X, 0]
        V = flow[Y, X, 1]

        # Quiver: Y is row, so invert it if your coordinate system differs.
        # Typically, image coords have Y increasing downward, so we might do -V if we want "up" in the positive direction.
        self.quiver = self.ax_img.quiver(
            X, Y, U, V,
            color='red',
            units='xy',
            angles='xy',
            scale_units='xy',

            # Make arrows bigger (smaller 'scale' => longer arrows)
            scale=0.05,

            # Pivot: 'tail' => (X,Y) is at arrow tail; 'mid' => arrow centered on (X,Y);
            #        'tip' => arrow tip is at (X,Y).
            pivot='tail',

            # Increase arrow thickness
            width=0.005,

            # Increase arrowhead size
            headwidth=10,
            headlength=12,
            headaxislength=8,

            # Optional: outline thickness and color
            linewidths=0.7,
            edgecolors='yellow', 

            alpha=1.0,  # fully opaque
            zorder=2    # drawn above the image
        )
        
        # Make sure the axis matches the image's dimension
        self.ax_img.set_xlim(0, W)
        self.ax_img.set_ylim(H, 0)  # Flip y so 0 is at the top


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    
    chart = MultiBollingerChart(t_window=10, num_std=2.0)

    # For demonstration, we'll update 2 lines: "LineA" and "LineB".
    line_names = ["LineA", "LineB"]

    for i in range(30):
        # Create some random Bollinger data
        data_dict = {}
        for ln in line_names:
            offset = {"LineA": 0, "LineB": 20}[ln]
            val = 100 + offset + random.gauss(0, 1)
            std = random.uniform(0.5, 2.0)
            data_dict[ln] = (val, std)

        # Create a random 100x150 grayscale image
        H, W = 100, 150
        frame = np.random.randint(0, 255, (H, W), dtype=np.uint8)

        # Create a random flow field (H, W, 2)
        # For real use, you'd have something from cv2.calcOpticalFlowFarneback(...)
        flow = np.random.randn(H, W, 2).astype(np.float32) * 2.0

        # Send everything to the chart at once
        chart.push_new_data(data_dict, frame=frame, flow=flow)

        time.sleep(0.5)

    input("Press Enter to exit...")
