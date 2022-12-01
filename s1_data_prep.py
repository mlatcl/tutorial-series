
import os

def plot_path(path, ion=True):

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    if ion:
        plt.ion()
        plt.style.use('seaborn-pastel')

    proj = ccrs.SouthPolarStereo()
    transform = ccrs.PlateCarree()

    fig = plt.figure()
    ax = fig.add_subplot(projection=proj)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.set_extent([-80, 0, -75, -65], transform)

    plt.scatter(
        x=path[:, 1], y=path[:, 0], # lon, lat
        s=2, color="dodgerblue", transform=transform
    )

if __name__ == "__main__" and not os.path.exists('1_data.pkl'):

    import numpy as np
    import xarray as xr
    import pickle as pkl
    from tqdm import tqdm

    file_dir = '1_raw_data'
    files = [os.path.join(file_dir, f) for f in os.listdir(file_dir)]
    files = np.array(files, dtype=str) # ERA5 sea ice concentration
    files.sort()

    template_data = xr.open_dataset(files[0])

    # using google maps, a path from Marguerite Bay to Weddell Sea
    path = np.array((
        (-67.85, -70.80),
        (-63.56, -66.29),
        (-62.30, -55.12),
        (-65.48, -38.63),
        (-71.33, -16.16),
    ))

    # linearly interpolated path with more points
    points = []
    for i in range(len(path) - 1):
        r = np.linspace(0, 1, 25)
        points.append(
            np.vstack([
                r*path[i + 1, 0] + (1-r)*path[i, 0],
                r*path[i + 1, 1] + (1-r)*path[i, 1]]).T
        )
    points = np.vstack(points)

    # extract indices in the template file closest to our path
    lats = template_data.lat.to_numpy()
    lons = template_data.lon.to_numpy()

    def closest_idx(i):
        return np.unravel_index(
            np.sqrt(np.square(lats - points[i, 0]) +
                    np.square(lons - points[i, 1])
                ).argmin(), lats.shape)

    idxs = [closest_idx(i) for i in range(len(points))]
    idxs_unique = []
    for idx_set in idxs:
        if idx_set not in idxs_unique:
            idxs_unique.append(idx_set)
    idxs = idxs_unique

    # closest points in file to path
    points_recomp = np.array([(lats[i, j], lons[i, j]) for (i, j) in idxs])

    # check if path looks ok
    plot_path(points_recomp)

    def load_concs(file):
        data = xr.open_dataset(file)
        return np.array([data.ice_conc.to_numpy()[0][i, j] for (i, j) in idxs])[None, :]

    # concentration data
    ice_concs = np.concatenate([load_concs(f) for f in tqdm(files)], axis=0)/100

    with open('1_data.pkl', 'wb') as file:
        pkl.dump((points_recomp, ice_concs), file)
