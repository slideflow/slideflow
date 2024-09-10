import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import slideflow as sf

from typing import Union, Tuple, Optional
from . import cwh_to_whc, whc_to_cwh, is_whc

# -----------------------------------------------------------------------------

class ColorProfile:
    """Color profile for histogram matching."""

    def __init__(self, mapping: np.ndarray) -> None:
        self.mapping = mapping.astype(np.uint8)
        self.mapping_tensor = torch.from_numpy(self.mapping).long()

    @classmethod
    def build(cls, arg1, arg2, apply_loess: bool = True):
        if isinstance(arg1, sf.WSI):
            obj = cls.build_from_slide(arg1, arg2)
        else:
            obj = cls.build_from_img(arg1, arg2)
        if apply_loess:
            obj.apply_loess()
        return obj

    @classmethod
    def build_from_slide(
        cls,
        wsi: "sf.WSI",
        wsi_reference: "sf.WSI",
        n: int = 100
    ):
        """Build a color profile from a slide and a reference slide."""
        mapping = []
        s1 = set([tuple(idx) for idx in np.column_stack(wsi.grid.nonzero())])
        s2 = set([tuple(idx) for idx in np.column_stack(wsi_reference.grid.nonzero())])
        all_idx = np.array(list(s1.intersection(s2)))
        for _ in range(n):
            idx = all_idx[np.random.choice(np.arange(all_idx.shape[0]))]
            img_target = wsi[idx[0], idx[1]]
            img_ref = wsi_reference[idx[0], idx[1]]
            matched_target = cls.match_histogram(img_target, img_ref)
            _mapping = cls.get_target_mapping(img_target, matched_target)
            mapping.append(_mapping)
        mapping = np.array(mapping)
        mapping = np.nanmean(mapping, axis=0)
        #mapping = cls.fit_from_mapping(mapping)
        return cls(mapping)

    @classmethod
    def build_from_img(cls, img: np.ndarray, img_reference: np.ndarray):
        matched_target = cls.match_histogram(img, img_reference)
        mapping = cls.get_target_mapping(img, matched_target)
        return cls(mapping)

    def apply_loess(self) -> None:
        """Apply a loess smoothing to the color mapping."""
        try:
            from skmisc.loess import loess
        except ImportError:
            raise ImportError("The 'skmisc' package is required for loess smoothing.")
        for c in range(3):
            nonzero_idx = np.array([0] + list(np.nonzero(self.mapping[c])[0]))
            nonzero_vals = self.mapping[c][nonzero_idx]
            ol = loess(nonzero_idx, nonzero_vals, span=0.1, surface='direct')
            ol.fit()
            pred = ol.predict(np.arange(256))
            interpolated = np.clip(pred.values, 0, 255).astype(int)

            # If we are interpolating values at the start or end, ensure that they do not change the curve direction.
            # At the end
            if nonzero_idx[-1] != 255:
                change_indices = np.where(np.diff(np.concatenate(([1], self.mapping[c] == 0, [1]))))[0]
                start_of_last_hole = change_indices[-1]
                _min = int(interpolated[start_of_last_hole-1])
                if start_of_last_hole == 255:
                    interpolated[start_of_last_hole] = min(_min, interpolated[start_of_last_hole])
                else:
                    interpolated[start_of_last_hole:] = np.clip(interpolated[start_of_last_hole:], _min, 255)
            # At the start
            if nonzero_idx[1] != 1:
                change_indices = np.where(np.diff(np.concatenate(([1], self.mapping[c] == 0, [1]))))[0]
                end_of_first_hole = change_indices[0]
                _max = int(interpolated[end_of_first_hole+1])
                if end_of_first_hole == 1:
                    interpolated[end_of_first_hole] = max(_max, interpolated[end_of_first_hole])
                else:
                    interpolated[:end_of_first_hole+1] = np.clip(interpolated[:end_of_first_hole+1], 0, _max)

            self.mapping[c] = interpolated

    @staticmethod
    def get_target_mapping(target: np.ndarray, matched_target: np.ndarray) -> np.ndarray:
        """Get the mapping from the target image to the matched target image."""
        x_red = target[:, :, 0].flatten()
        y_red = matched_target[:, :, 0].flatten()
        yz_red = np.zeros(256)
        for n in range(0, 256):
            yz_red[n] = y_red[np.argwhere(x_red == n)].mean()

        x_green = target[:, :, 1].flatten()
        y_green = matched_target[:, :, 1].flatten()
        yz_green = np.zeros(256)
        for n in range(0, 256):
            yz_green[n] = y_green[np.argwhere(x_green == n)].mean()

        x_blue = target[:, :, 2].flatten()
        y_blue = matched_target[:, :, 2].flatten()
        yz_blue = np.zeros(256)
        for n in range(0, 256):
            yz_blue[n] = y_blue[np.argwhere(x_blue == n)].mean()
        return np.stack([yz_red, yz_green, yz_blue])

    @staticmethod
    def match_histogram(target_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
        """Adjust the target image to match the histogram of the reference image.

        Args:
            target_img (np.ndarray): Target image (to be transformed)
            reference_img (np.ndarray): Reference image.

        Returns:
            np.ndarray: Histogram-matched image
        """
        # Split the images into their respective channels
        target_img_channels = cv2.split(target_img)
        reference_img_channels = cv2.split(reference_img)
        matched_channels = []

        for target_channel, reference_channel in zip(target_img_channels, reference_img_channels):
            # Compute the histograms and their cumulative distributions
            target_hist, _ = np.histogram(target_channel.flatten(), 256, [0, 256])
            target_cdf = target_hist.cumsum()
            target_cdf_normalized = target_cdf / float(target_cdf.max())

            reference_hist, _ = np.histogram(reference_channel.flatten(), 256, [0, 256])
            reference_cdf = reference_hist.cumsum()
            reference_cdf_normalized = reference_cdf / float(reference_cdf.max())

            # Create a lookup table to map target pixels to match the reference histogram
            lookup_table = np.zeros(256)
            reference_value = 0
            for target_value in range(256):
                while reference_cdf_normalized[reference_value] < target_cdf_normalized[target_value] and reference_value < 255:
                    reference_value += 1
                lookup_table[target_value] = reference_value

            # Map the target channel pixels using the lookup table
            matched_channel = np.interp(target_channel.flatten(), np.arange(256), lookup_table)
            matched_channels.append(matched_channel.reshape(target_channel.shape).astype(np.uint8))

        # Merge the channels back together
        matched_img = cv2.merge(matched_channels)
        return matched_img

    def plot(self) -> None:
        plt.scatter(range(256), self.mapping[0], s=1, color='red')
        plt.scatter(range(256), self.mapping[1], s=1, color='green')
        plt.scatter(range(256), self.mapping[2], s=1, color='blue')
        plt.xlabel('Original intensity')
        plt.ylabel('Intensity after histogram match')

    def apply(
        self,
        img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply the color mapping to an image.

        Args:
            img (Union[np.ndarray, torch.Tensor]): Image to apply the color mapping to.

        Returns:
            Union[np.ndarray, torch.Tensor]: Image with the color mapping applied.

        """
        if isinstance(img, np.ndarray):
            return self.apply_numpy(img)
        elif isinstance(img, torch.Tensor):
            return self.apply_torch(img)

    def apply_numpy(self, img):
        return np.stack([
            self.mapping[0, img[:, :, 0]],  # Apply mapping for Red channel
            self.mapping[1, img[:, :, 1]],  # Apply mapping for Green channel
            self.mapping[2, img[:, :, 2]]   # Apply mapping for Blue channel
        ], axis=-1)

    def apply_torch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply color mapping to a batch of images using PyTorch.

        Args:
            images (torch.Tensor): A PyTorch tensor of shape (batch_size, 3, height, width) representing a batch of images.

        Returns:
            torch.Tensor: A PyTorch tensor of the same shape as `images` with the color mapping applied.
        """
        img_is_whc = is_whc(images)
        if img_is_whc:
            images = whc_to_cwh(images)
        if images.ndim == 3:
            is_single_image = True
            images = torch.unsqueeze(images, 0)
        else:
            is_single_image = False

        # Ensure the images and mapping are on the same device and mapping is of type long for indexing
        mapping = self.mapping_tensor.to(images.device)
        images = images.long()

        # Apply the mapping to each channel
        mapped_images = torch.stack([
            mapping[0][images[:, 0, :, :]],  # Red channel
            mapping[1][images[:, 1, :, :]],  # Green channel
            mapping[2][images[:, 2, :, :]]   # Blue channel
        ], dim=1)  # Stack along the channel dimension to maintain (batch_size, 3, height, width) shape
        mapped_images = mapped_images.to(torch.uint8)

        if is_single_image:
            mapped_images = torch.squeeze(mapped_images, 0)
        if img_is_whc:
            return cwh_to_whc(mapped_images)
        else:
            return mapped_images


# -----------------------------------------------------------------------------

def generate_bezier_points(p0: int, p1: int, p2: int, p3: int, t_values: np.ndarray) -> np.ndarray:
    """
    Generate Bezier curve points for a given segment.
    """
    t = t_values[:, None]  # Convert to column vector for broadcasting
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def random_handle_length() -> int:
    """Generate a random handle length within the specified range."""
    return np.random.uniform(15, 50)


def slope_to_dx(angle_degrees: int) -> np.ndarray:
    angle_radians = np.radians(angle_degrees)
    dx = np.cos(angle_radians)
    dy = np.sin(angle_radians)
    return np.column_stack([dx, dy])


def random_slope() -> np.ndarray:
    """
    Generate a random slope by specifying an angle in degrees.
    Returns the slope as a unit vector (dx, dy).
    """
    angle_degrees = np.random.uniform(35, 60)  # Random angle between -45 and 45 degrees
    return slope_to_dx(angle_degrees)


def find_segment(x: int, segments: np.ndarray) -> int:
    """Find which segment an x-value belongs to """
    for i, (start, end) in enumerate(segments):
        if start <= x <= end:
            return i
    return -1


def simulate_adjusted_bezier_handles_mapped(
    x: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    slopes: np.ndarray
):
    """
    Simulate a Bezier curve with adjusted handles to match the given control points.

    Args:
        x (np.ndarray): The x-coordinates of the control points.
        y (np.ndarray): The y-coordinates of the control points.
        lengths (np.ndarray): The lengths of the handles.
        slopes (np.ndarray): The slopes of the handles.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The x and y coordinates of the simulated curve.
    """
    assert len(x) == len(y), "x and y must have the same length."

    # Prepare the x values for which we want to find the corresponding y values
    x_mapped = np.arange(256)
    y_mapped = np.zeros_like(x_mapped)

    # Vectorized calculations
    segments_start = x[:-1]
    segments_end = x[1:]
    segments_mask = np.logical_and(x_mapped >= segments_start[:, np.newaxis], x_mapped < segments_end[:, np.newaxis])
    segment_indices = np.argmax(segments_mask, axis=0)

    p0 = np.column_stack((x[:-1], y[:-1]))
    p3 = np.column_stack((x[1:], y[1:]))

    handle_lengths_start = lengths[:-1, 0]
    handle_lengths_end = lengths[:-1, 1]

    slopes_start = slope_to_dx(slopes[:-1, 0])
    slopes_end = slope_to_dx(slopes[:-1, 1])

    p1 = p0 + handle_lengths_start[:, np.newaxis] * slopes_start
    p2 = p3 - handle_lengths_end[:, np.newaxis] * slopes_end

    t = (x_mapped - segments_start[segment_indices]) / (segments_end[segment_indices] - segments_start[segment_indices])

    t_powers = np.column_stack((
        (1 - t) ** 3,
        3 * (1 - t) ** 2 * t,
        3 * (1 - t) * t ** 2,
        t ** 3
    ))

    bezier_points = np.stack((p0[segment_indices], p1[segment_indices], p2[segment_indices], p3[segment_indices]), axis=1)
    y_mapped = np.sum(t_powers[:, :,] * bezier_points[:, :, 1], axis=1)

    return x_mapped, y_mapped.clip(0, 255)


def generate_random_curve(
    n_points: Optional[int] = None,
    x_control: Optional[np.ndarray] = None,
    y_control: Optional[np.ndarray] = None,
    slope_range: Tuple[int, int] = (30, 60),
    plot: bool = False,
    color: str = 'blue'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random curve by simulating a Bezier curve with random control points.

    Args:
        n_points (int, optional): The number of random control points to generate.
            If not supplied, will randomly choose 0-3.
        x_control (np.ndarray, optional): The x-coordinates of the control points.
        y_control (np.ndarray, optional): The y-coordinates of the control points.
        slope_range (Tuple[int, int], optional): The range of slopes to use for the handles.
            Defaults to (30, 60).
        plot (bool, optional): Whether to plot the curve. Defaults to False.
        color (str, optional): The color to use for the plot. Defaults to 'blue'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The x and y coordinates of the generated curve.
    """
    if n_points is None:
        n_points = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
    if x_control is None:
        x_control = np.array(sorted([0, 255] + [np.random.uniform(0, 255) for _ in range(n_points)]))
    if y_control is None:
        y_control = np.array(sorted([0, 255] + [np.random.uniform(0, 255) for _ in range(n_points)]))
    lengths = []
    slopes = []

    # Starting point
    for i in range(len(x_control)):
        if i == len(x_control) - 1:
            dist_to_next = 127
        else:
            dist_to_next = x_control[i+1] - x_control[i]
        lengths.append([np.random.uniform(0, dist_to_next//2), np.random.uniform(0, dist_to_next//2)])
        slopes.append([np.random.uniform(*slope_range), np.random.uniform(*slope_range)])

    lengths = np.array(lengths)
    slopes = np.array(slopes)

    x_curve, y_curve = simulate_adjusted_bezier_handles_mapped(x_control, y_control, lengths, slopes)

    # Optionally, plot the curve to see what it looks like
    if plot:
        plt.scatter(x_curve, y_curve, label='Bezier Curve', s=1, color=color)
        plt.scatter(x_control, y_control, color=color, label='Control Points')
        plt.legend()

    return x_curve, y_curve


def generate_constrained_curve(
    x_base_curve: np.ndarray,
    y_base_curve: np.ndarray,
    max_y_dist: int = 20,
    **kwargs
) -> np.ndarray:
    """
    Generate a random curve that is constrained to be within a certain distance of a base curve.
    """

    n_points = np.random.choice([0, 1, 2, 3], p=[0.5, 0.35, 0.1, 0.05])
    x_control = np.array(sorted([0, 255] + [np.random.uniform(0, 255) for _ in range(n_points)]))
    y_control = [0]
    for i in range(n_points):
        x = int(x_control[i+1])
        y = np.random.uniform(y_base_curve[x]-max_y_dist, y_base_curve[x]+max_y_dist)
        y_control.append(y)
    y_control.append(255)
    y_control = np.array(sorted(y_control))
    return generate_random_curve(x_control=x_control, y_control=y_control, **kwargs)


def generate_random_color_mapping(plot: bool = False, **kwargs) -> np.ndarray:
    """
    Generate a random color mapping by simulating a Bezier curve with random control points.
    """

    # Generate the base curve.
    _x = np.random.uniform(55, 200)
    _y = 255 - _x
    x_control = np.array([0, _x, 255])
    y_control = np.array([0, _y, 255])
    lengths = []
    slopes = []
    base_slope = np.random.uniform(35, 55)

    # Starting point
    for i in range(len(x_control)):
        if i == len(x_control) - 1:
            dist_to_next = 127
        else:
            dist_to_next = x_control[i+1] - x_control[i]
        lengths.append([np.random.uniform(0, dist_to_next//2), np.random.uniform(0, dist_to_next//2)])
        slopes.append([base_slope, base_slope])

    lengths = np.array(lengths)
    slopes = np.array(slopes)

    x_base_curve, y_base_curve = simulate_adjusted_bezier_handles_mapped(x_control, y_control, lengths, slopes)

    if plot:
        plt.scatter(x_base_curve, y_base_curve, label='Bezier Curve', s=1, color='black')
        plt.scatter(x_control, y_control, color='black', label='Control Points')
        plt.legend()

    r = generate_constrained_curve(x_base_curve, y_base_curve, color='red', plot=plot, **kwargs)[1]
    g = generate_constrained_curve(x_base_curve, y_base_curve, color='green', plot=plot, **kwargs)[1]
    b = generate_constrained_curve(x_base_curve, y_base_curve, color='blue', plot=plot, **kwargs)[1]
    return np.stack([r, g, b])
