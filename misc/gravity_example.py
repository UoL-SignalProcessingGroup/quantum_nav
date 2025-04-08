"""
==================
gravity_example.py
==================

:summary:
    This module provides example usage of the gravity models included in
    the toolbox. At a minimum, gravity is represented by a base gravity model,
    which is an approximation formula (such as the Somigliana formula). An
    optional extension to the selected formula in the form of a gravity
    correction map can be applied. These use the base values produced by the
    base formula and interpolate gravity anomaly or disturbance measures to
    produce corrected gravity values. The correction maps supported are in
    various formats and resolutions. They are also location limited, meaning
    at times the base formula will be used. The list of supported gravity
    models and maps are as follows:

    Base Gravity Formulas:
    -   FixedValue (9.81 always constant)
    -   SimpleUniform (Simple uniform sphere)
    -   Somigliana (with height interpolation)
    -   WGS84Gravity (NIMA's WGS84 model)

    Gravity Correction Maps:
    -   Geosat44 (Scatter point anomaly correction)
    -   MarineGravity (Grid-Base anomaly correction)
    -   GGMPlusAcc (GGMPlus acceleration correction)
    -   GGMPlusDist (GGMPlus disturbance correction)
    -   SRTM2GravityRes (SRTM2 gravity residual correction)
    -   SRTM2GravityFull (SRTM2 gravity full-scale correction)
    -   IrishSea (Bespoke Irish sea gravity anomaly correction)
"""

# For large number of points
import numpy as np

# Import all gravity base formulas
from qnav.gravity.simple import FixedValue
from qnav.gravity.simple import SimpleUniform
from qnav.gravity.somigliana import Somigliana
from qnav.gravity.nima import WGS84Gravity

# Import all geoid models
from qnav.earth.geoid import GeoidPGM

# Import all gravity correction maps
from qnav.gravity.map import CircularGrid
from qnav.gravity.geosat import Geosat44
from qnav.gravity.marine import MarineGravity
from qnav.gravity.irish import IrishSeaGravity
from qnav.gravity.ggm_plus import GGMPlusAcc, GGMPlusDist
from qnav.gravity.srtm2gravity import SRTM2GravityRes, SRTM2GravityFull

# Import path handling modules
from pathlib import Path


# If the script is being run directly:
if __name__ == '__main__':

    """
    Step 1: Set up database directories
            This is where the data is to be downloaded and stored to.
            
    """
    # Create a temporary directory
    # import tempfile
    # project_dir = tempfile.mkdtemp('qnav')

    # Provide paths for where to store data.
    project_dir = Path(__file__).parent
    geoid_dir = project_dir / 'databases' / 'geoid'
    gravity_dir = project_dir / 'databases' / 'gravity'


    """
    Step 2: Select a base gravity model
            This will provide an base approximation for acceleration.
            Recommended to use either Somigliana or WGS84Gravity.
    """
    # Select the base gravity function.
    # gravity_func = FixedValue()
    # gravity_func = SimpleUniform()
    gravity_func = Somigliana()
    # gravity_func = WGS84Gravity()


    """
    Step 3: Select a geoid model
            This is optional and depends on the correction map.
            Resolutions are given in arc-minutes (lower = larger).
            Setting as None will assume all geoid heights are zero.
    """
    # Select a geoid model.
    # geoid_model = GeoidPGM(geoid_dir / 'egm84-30.pgm')
    # geoid_model = GeoidPGM(geoid_dir / 'egm84-15.pgm')
    # geoid_model = GeoidPGM(geoid_dir / 'egm96-15.pgm')
    # geoid_model = GeoidPGM(geoid_dir / 'egm96-5.pgm')
    geoid_model = GeoidPGM(geoid_dir / 'egm2008-5.pgm')
    # geoid_model = GeoidPGM(geoid_dir / 'egm2008-2_5.pgm')
    # geoid_model = GeoidPGM(geoid_dir / 'egm2008-1.pgm')
    # geoid_model = None


    """
    Step 4: Select a gravity correction map.
            This will handle interpolating values from a correction map.
            Models are given in order of increasing resolution size (except
            for Irish sea). Some models take some time to download and set-up. 
            The GGMPlus and SRTM2Gravity models only load the necessary data 
            on demand. 
    """
    auto_download = True  # Set to true to download missing files automatically.
    tile_margin = 1       # The number of surrounding cache tiles to load.

    # gravity_model = IrishSeaGravity(gravity_func, geoid_model, geoid_dir / 'irish_sea')
    # gravity_model = Geosat44(gravity_func, geoid_model, geoid_dir / 'geosat', auto_download)
    # gravity_model = MarineGravity(gravity_func, geoid_model, geoid_dir / 'marine', auto_download)
    # gravity_model = GGMPlusAcc(gravity_func, geoid_model, geoid_dir / 'ggm_plus', auto_download, tile_margin)
    # gravity_model = GGMPlusDist(gravity_func, geoid_model, geoid_dir / 'ggm_plus', auto_download, tile_margin)
    gravity_model = SRTM2GravityRes(gravity_func, geoid_model, geoid_dir / 'srtm2gravity', auto_download, tile_margin)
    # gravity_model = SRTM2GravityFull(gravity_func, geoid_model, geoid_dir / 'srtm2gravity', auto_download, tile_margin)


    """
    Step 5: Calculate some basic gravity values
            For given positions calculate the gravity acceleration. 
            Vectorisation helps when calculating for many positions simultaneously.
    """
    # Present a single test location (lat-lon-alt).
    test_position = [53.406520, -2.966922, 100.0]
    print(f'Test position (lat-lon-alt):\t {test_position}\n')

    # Calculate the gravity acceleration.
    g_z = gravity_model.calc_gravity_z(*test_position)
    print(f'Total gravity acceleration:\t\t {g_z:.8e}')

    # Calculate the gravity acceleration vector.
    g_xyz = gravity_model.calc_gravity_xyz(*test_position)
    print(f'Gravity acceleration vector:\t {g_xyz}\n')

    # Present multiple test locations (rows: lat-lon-alt).
    centre = np.array(test_position)
    random_offsets = np.random.randn(3, 1000) * 1e-4
    test_positions = centre[:, None] + random_offsets

    # Calculate the above, but for multiple points simultaneously.
    g_z_vec = gravity_model.calc_gravity_z_vec(*test_positions)
    g_xyz_vec = gravity_model.calc_gravity_xyz_vec(*test_positions).T  # By default columns are returned


    """
    Step 6: Calculate the vertical gravity gradient. 
            Gravity correction maps require surrounding grid points. 
    """
    # Construct a circular grid for sampling from the surrounding data.
    # Allows the calculation to use a local area to sample from.
    circular_grid = CircularGrid(
        num_rings=100,      # The number of surrounding layers
        num_angles=50,      # The number of points per a layer
        max_radius=0.25,    # The maximum distance to cover (in decimal degrees)
        l_threshold=1.0     # The margin to avoid around the centre (in metres).
    )

    # circular_grid = CircularGrid(
    #     num_rings=300,      # The number of surrounding layers
    #     num_angles=100,      # The number of points per a layer
    #     max_radius=0.25,    # The maximum distance to cover (in decimal degrees)
    #     l_threshold=2000     # The margin to avoid around the centre (in metres).
    # )

    # Calculate the approximated gravity gradient using just the formula.
    estimated_dgz_dz = gravity_func.calc_vertical_grad(*test_position)
    print(f'Estimated gravity gradient:\t\t {estimated_dgz_dz:.8e}')

    # Calculate the correct gravity gradient using the gravity map correction.
    corrected_dgz_dz = gravity_model.calc_vertical_grad(*test_position, circular_grid=circular_grid)
    print(f'Corrected gravity gradient:\t\t {corrected_dgz_dz:.8e}')

    import time
    start = time.time()
    corrected_dgz_dz = gravity_model.calc_vertical_grad(
        *test_position, circular_grid=circular_grid)
    end = time.time()
    print(f'\nTime elapsed: {end - start:.8f} s')
