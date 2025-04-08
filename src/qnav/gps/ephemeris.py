"""
============
ephemeris.py
============

:summary:
    Collection of functions for extracting GPS/GNSS information.
    This module is responsible for the util satellite functionality within
    the toolbox. This includes functions for downloading GPS information,
    parsing RINEX files, extracting satellite information and estimating
    missing values.

:authors:
    | Michael Wright - mjwright@liverpool.ac.uk
    | Dr. Niall Moroney - niall.moroney17@imperial.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""


# .. TODO:
#       - Add better method for holding Rinex records
#       - Support more RINEX file versions that just 3.03
#       - Read version from header and raise error if unsupported


import numpy as np
import ftplib
import math
import gzip

from tempfile import NamedTemporaryFile
from numpy.linalg import norm
from datetime import datetime
from datetime import timedelta
from ftplib import FTP_TLS
from enum import StrEnum
from pathlib import Path
from copy import copy

from qnav.gps.satellite import SatelliteParams
from qnav.gps.satellite import satellite_param_fields
from qnav.util.constants import OMEGA_E
from qnav.util.constants import GM


#: The FTP server address for downloading RINEX files.
__FTP_ADDR = 'gdc.cddis.eosdis.nasa.gov'

#: The FTP date formatting string for hourly data directories.
__FTP_DIR_DATE_STR = '/gnss/data/hourly/%Y/%j/%H'

#: The suffix/extension to match with files to download from.
__FPT_FILE_SUFFIX = 'MN.rnx.gz'


class SatelliteType(StrEnum):
    """
    Character identifiers for the different satellite types.
    These characters identify the record types corresponding to the satellite
    systems supported inside a Rinex File (as of version 4).
    """
    GPS = 'G',
    GLONASS = 'R',
    GALILEO = 'E',
    QZSS = 'J',
    BDS = 'C',
    IRNSS = 'I',
    SBAS = 'S'
    MIXED = 'M'


def get_rinex_path(gps_time: datetime, directory: Path) -> Path:
    """
    Obtain the location of a local RINEX file identified by GPS date and time.
    This function will return the local filepath of a RINEX file corresponding
    to a given (python) datetime. The file of the returned path may or may not
    exist (requiring downloading if missing).

    :param gps_time: Python datetime object for the time required for the ephemeris.
    :type gps_time: datetime.datetime

    :param directory: Location for the files to be downloaded to.
    :type directory: Path

    :return: The path of the corresponding RINEX file.
    :rtype: Path
    """

    eph_year = gps_time.strftime('%Y')
    eph_file = gps_time.strftime("%Y_%j.rnx")
    return directory / eph_year / eph_file


def download_rinex_file(gps_time: datetime, database_dir: Path):
    """
    This function downloads and extracts an ephemeris file for a specified
    day from "cddis.gsfc.nasa.gov" and saves it under the provided directory.

    :param gps_time: The Python datetime object for the time required for the
        ephemeris data.
    :type: datetime.datetime

    :param database_dir: The location for the files to be downloaded to.
    :type database_dir: Path
    """

    # Get the expected parent directory of the on the NASA server
    fpt_dir = gps_time.strftime(__FTP_DIR_DATE_STR)

    # Get the local file paths for the download destinations
    eph_file_path = get_rinex_path(gps_time, database_dir)

    # If unzipped ephemeris file does not exist, download and unpack it:
    if not eph_file_path.exists():

        # Create any missing directories:
        if not eph_file_path.parent.is_dir():
            eph_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with FTP_TLS(__FTP_ADDR) as ftp:

                ftp.login()
                ftp.prot_p()

                ftp.cwd(fpt_dir)
                files = []
                ftp.dir(files.append)

                files = ([list(filter(None, file.split(" "))) for
                          file in files if file.endswith(__FPT_FILE_SUFFIX)])

                sizes = [int(file[4]) for file in files]
                file = files[sizes.index(max(sizes))][-1]

                # Create any missing directories:
                if not database_dir.is_dir():
                    database_dir.mkdir(parents=True, exist_ok=True)

                # Generate a temporary file to hold downloaded data:
                with NamedTemporaryFile(dir=database_dir, suffix='.Z') as tmp:

                    # Write the data to the temporary file:
                    ftp.retrbinary('RETR ' + file, tmp.write)
                    tmp.flush()
                    tmp.seek(0)

                    # Extract and write the decompressed content:
                    with (gzip.GzipFile('rb', fileobj=tmp) as input_file,
                            open(eph_file_path, 'wb') as output_file):
                        output_file.write(input_file.read())

        except ftplib.error_perm:
            raise FileNotFoundError(
                f'Unable to acquire GPS ephemeris file for {str(gps_time.date())}')

        except ConnectionError as con_e:
            raise ConnectionError(
                f'Unable to download GPS ephemeris file\n{str(con_e)}')

        except Exception as e:
            raise e


def download_rinex_files(from_time: datetime,
                         to_time: datetime,
                         database_dir: Path,
                         inc_last: bool = True):
    """
    This function downloads and extracts multiple ephemeris files.
    Given a date range, this function will download all the ephemeris
    files covered by the range.

    :param from_time: The first date for the download range.
        This is the date for the first GPS file to be downloaded.
    :type from_time: datetime.datetime

    :param to_time: The last date for the download range.
        This is the end date for the GPS files to download.
    :type to_time: datetime.datetime

    :param database_dir: The location for the files to be downloaded to.
    :type database_dir: Path

    :param inc_last: Should the last date of the range be included?
    :type inc_last: bool
    """

    # Create any missing directories:
    if not database_dir.is_dir():
        database_dir.mkdir(parents=True, exist_ok=True)

    with FTP_TLS(__FTP_ADDR) as ftp:

        ftp.login()
        ftp.prot_p()

        # Extract the date range requested
        gps_time = copy(from_time)
        end_date = to_time
        num_days = (end_date - gps_time).days

        # Download the GPS data for all the dates given:
        for i in range(num_days + inc_last):

            # Print batch downloading status in console:
            print(f"Downloading GPS Data {i+1} of {num_days+1}:\t",
                  gps_time.strftime("%a %d %b %Y"))

            # Get the expected parent directory of the on the NASA server
            fpt_dir = gps_time.strftime(__FTP_DIR_DATE_STR)

            # Get the local file paths for the download destinations
            eph_file_path = get_rinex_path(gps_time, database_dir)

            # Create any missing directories:
            if not eph_file_path.parent.is_dir():
                eph_file_path.parent.mkdir(parents=True, exist_ok=True)

            # If unzipped ephemeris file does not exist, download and unpack it:
            if not eph_file_path.exists():

                ftp.cwd('/')
                ftp.cwd(fpt_dir)

                files = []
                ftp.dir(files.append)
                files = ([list(filter(None, file.split(" "))) for
                          file in files if file.endswith(__FPT_FILE_SUFFIX)])

                sizes = [int(file[4]) for file in files]
                file = files[sizes.index(max(sizes))][-1]

                # Generate a temporary file to hold downloaded data:
                with NamedTemporaryFile(dir=database_dir, suffix='.Z') as tmp:

                    # Write the data to the temporary file:
                    ftp.retrbinary('RETR ' + file, tmp.write)
                    tmp.flush()
                    tmp.seek(0)

                    # Extract and write the decompressed content:
                    with (gzip.GzipFile('rb', fileobj=tmp) as input_file,
                          open(eph_file_path, 'wb') as output_file):
                        output_file.write(input_file.read())

            # Increment the time to the next day:
            gps_time += timedelta(hours=24)


# def read_ephemeris_file(ephemeris_file: Path) -> list[SatelliteParams]:
#     """
#     Parses a downloaded ephemeris file and returns its data.
#     Reads a given ephemeris file, skipping the header contents and returning
#     the contained satellite parameter records
#
#     :param ephemeris_file: The local ephemeris file to read.
#     :type ephemeris_file: Path
#
#     :returns: A list of extracted satellite parameters for GPS satellites.
#     :rtype: list[SatelliteParams]
#     """
#
#     # https://files.igs.org/pub/data/format/rinex303.pdf
#
#     satellite_ephemeris = dict()
#     satellite_list = []
#
#     # Read File and ignore header
#     with open(ephemeris_file, 'r') as file:
#
#         line = file.readline()
#
#         while line.find("END OF HEADER") == -1:
#             line = file.readline()
#
#         # Read blocks of four lines, each specifying one satellite
#         prefix_length: dict[str, int] = {
#             'G': 8, 'R': 5, 'S': 4, 'E': 8, 'C': 8, 'J': 8
#         }
#
#         # Conversion for code to satellite description
#         prefix_type: dict[str, str] = {
#             'G': 'GPS', 'R': 'GLONASS', 'S': 'SBAS',
#             'E': 'Galileo', 'C': 'BeiDou', 'J': 'QZSS'
#         }
#
#         line = file.readline().replace('D', 'E')
#         while line:
#
#             line_count = 1
#             block: dict[str, any] = dict()
#
#             length = prefix_length[line[0]]
#             block['type'] = prefix_type[line[0]]
#             block['id'] = int(line[1:3])
#             block['year'] = int(line[4:8])
#             block['month'] = int(line[9:11])
#             block['day'] = int(line[12:14])
#             block['time'] = line[15:23]
#             block['clock_bias'] = float(line[23:42])
#             block['drift_freq_bias'] = float(line[42:61])
#             block['drift_rate_message_time'] = float(line[61:80])
#             ephemeris_data = []
#
#             while line_count < length:
#                 line_count += 1
#                 line = file.readline().replace('D', 'E')
#                 items = [line[4:23], line[23:42], line[42:61], line[61:80]]
#                 ephemeris_data += items
#
#             if length == 4:
#                 block['t_0e'] = block['drift_rate_message_time']
#                 block['prn'] = block['id']
#                 block['delta_n'] = 0
#                 block['i_dot'] = 0
#                 block['c_us'] = 0
#                 block['c_uc'] = 0
#                 block['c_is'] = 0
#                 block['c_ic'] = 0
#                 block['c_rs'] = 0
#                 block['c_rc'] = 0
#                 block['omega_dot'] = 0
#
#                 r_pz90f = np.array([
#                     float(ephemeris_data[0]),
#                     float(ephemeris_data[4]),
#                     float(ephemeris_data[8])
#                 ])
#
#                 v_pz90f = np.array([
#                     float(ephemeris_data[1]),
#                     float(ephemeris_data[5]),
#                     float(ephemeris_data[9])
#                 ])
#
#                 time_pz90f = datetime(
#                     block['year'], block['month'], block['day'],
#                     int(block['time'][0:2]), int(block['time'][3:5]), int(block['time'][6:9]))
#
#                 # Convert given satellite position and velocity into ECI frame
#                 r_eci, v_eci = _pz90f2eci(time_pz90f, r_pz90f, v_pz90f)
#
#                 # Convert this position and velocity into Kepler orbital
#                 # parameters
#                 kep_params = _get_kepler_parameters(r_eci, v_eci)
#
#                 # Assign Kepler parameters
#                 block['sqrt_a'] = (kep_params['a'] * 1000) ** 0.5
#                 block['e'] = kep_params['e']
#                 block['i_0'] = kep_params['i']
#                 block['omega_0'] = kep_params['Omega']
#                 block['w'] = kep_params['omega_small']
#                 block['m_bar_0'] = kep_params['theta']
#
#             else:
#                 # block['messageTime'] = block['drift_rate_message_time']
#                 block['sqrt_a'] = _float(ephemeris_data[7])
#                 block['e'] = _float(ephemeris_data[5])
#                 block['i_0'] = _float(ephemeris_data[12])
#                 block['omega_0'] = _float(ephemeris_data[10])
#                 block['w'] = _float(ephemeris_data[14])
#                 block['m_bar_0'] = _float(ephemeris_data[3])
#                 block['delta_n'] = _float(ephemeris_data[2])
#                 block['omega_dot'] = _float(ephemeris_data[15])
#                 block['i_dot'] = _float(ephemeris_data[16])
#                 block['c_us'] = _float(ephemeris_data[6])
#                 block['c_uc'] = _float(ephemeris_data[4])
#                 block['c_is'] = _float(ephemeris_data[11])
#                 block['c_ic'] = _float(ephemeris_data[9])
#                 block['c_rs'] = _float(ephemeris_data[1])
#                 block['c_rc'] = _float(ephemeris_data[13])
#                 # block['t_tx'] = _float(ephemeris_data[25])
#                 # block['GPS_Week'] = _float(ephemeris_data[18])
#                 block['t_0e'] = _float(ephemeris_data[8])
#                 block['prn'] = block['id']
#
#             sat_id = block['type'] + str(block['prn'])
#
#             if sat_id not in satellite_list and sat_id[0] not in ['S']:
#                 satellite_ephemeris[sat_id] = block
#                 # satellite_list.append(sat_id)
#
#             # satellite_ephemeris[sat_id] = block
#             line = file.readline().replace('D', 'E')
#
#     to_return = []
#     for k, v in satellite_ephemeris.items():
#         if len(k) > 2 and k[0:3] == 'GPS':
#             to_return.append(SatelliteParams(**v))
#
#     return to_return


def read_rinex_file(ephemeris_file: Path) -> dict[str, dict]:
    """
    Parses a downloaded ephemeris file and returns its data.
    Reads a given ephemeris file, skipping the header contents and returning
    the contained entry records in dictionaries. Each is keyed by the ID of
    the satellite.

    :param ephemeris_file: The local ephemeris file to read.
    :type ephemeris_file: Path

    :returns: A dictionary of extracted satellite parameters.
    :rtype: dict[str, dict]
    """

    file_data: dict[str, dict] = {}

    # Read file and ignore header
    with open(ephemeris_file, 'r') as file:

        header_suffix = 'END OF HEADER'
        line = file.readline()
        
        while line.find(header_suffix) == -1:
            line = file.readline()

        while line := file.readline():

            sat_id: str = line[:3]
            lines: list[str] = [line,]

            match sat_id[0]:

                case SatelliteType.GPS:
                    lines.extend([file.readline() for _ in range(7)])
                    file_data[sat_id] = _parse_gps_entry(lines)

                case SatelliteType.GLONASS:
                    lines.extend([file.readline() for _ in range(3)])  # Sometimes 4 lines!
                    file_data[sat_id] = _parse_gnss_entry(lines)

                case SatelliteType.GALILEO:
                    lines.extend([file.readline() for _ in range(7)])
                    file_data[sat_id] = _parse_galileo_entry(lines)

                case SatelliteType.QZSS:
                    lines.extend([file.readline() for _ in range(7)])
                    file_data[sat_id] = _parse_qzss_entry(lines)

                case SatelliteType.BDS:
                    lines.extend([file.readline() for _ in range(7)])
                    file_data[sat_id] = _parse_bds_entry(lines)

                case SatelliteType.IRNSS:
                    lines.extend([file.readline() for _ in range(7)])
                    file_data[sat_id] = _parse_irnss_entry(lines)

                case SatelliteType.SBAS:
                    lines.extend([file.readline() for _ in range(3)])
                    file_data[sat_id] = _parse_sbas_entry(lines)

                case ' ':
                    pass  # Skip until matching ID is found.

                case _:
                    raise KeyError(f'unsupported satellite ID \'{sat_id}\'')

    return file_data


# def rinex_to_satellite_params(rinex_data: dict, id_filter: str = r'.*') -> list[SatelliteParams]:
#
#     to_return: list[SatelliteParams] = []
#
#     for sat_id, records in rinex_data.items():
#         if re.match(id_filter, sat_id):
#             temp_dict = {k: v for k, v in records.items()
#                          if k in satellite_param_fields}
#             to_return.append(SatelliteParams(**temp_dict))
#
#     return to_return


def rinex_to_satellite_params(rinex_data: dict, *satellite_types: SatelliteType) -> list[SatelliteParams]:
    """
    Extracts given rinex data and converts into satellite parameters class.
    A helper function for extracting the usable satellite parameter data
    from given Rinex data, allowing for satellite instances to be created.
    Optional filtering can also be supplied to limit returns to specific
    satellite types.

    :param rinex_data: The data records read from a Rinex file.
    :type rinex_data: dict[str, dict]

    :param satellite_types: (Optional) The satellite types to filter by.
        By default, all types are selected and no filtering is performed.
    :type satellite_types: SatelliteType

    :return: The satellite parameters for each rinex record.
    :rtype: list[SatelliteParams]
    """

    # The satellite parameter list to return.
    to_return: list[SatelliteParams] = []

    # If no filtering is given, assume all
    if len(satellite_types) == 0:
        satellite_types = [s for s in SatelliteType]

    # For each satellite record :
    for sat_id, records in rinex_data.items():
        if sat_id[0] in satellite_types:

            # Extract keys that match the required.
            temp_dict = {k: v for k, v in records.items()
                         if k in satellite_param_fields}

            # Create satellite parameters and append to list
            to_return.append(SatelliteParams(**temp_dict))

    # Finally, return the satellite list.
    return to_return


def _parse_epoch_fields(line: str) -> dict:
    return {
        'type': line[0],
        'id': int(line[1:3]),
        'year': int(line[4:8]),
        'month': int(line[9:11]),
        'day': int(line[12:14]),
        'hour': int(line[15:17]),
        'minute': int(line[18:20]),
        'second': int(line[21:23]),
        'time': (line[15:23]),  # TODO: Fix ?
        'prn': int(line[1:3]),
    }


def _parse_gps_entry(lines: list[str]) -> dict[str, any]:

    epoch = _parse_epoch_fields(lines[0])
    record_fields = _parse_record_fields(lines)

    # Satellite System Records (see page 68)
    record_keys = [
        'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
        'iod_eph', 'c_rs', 'delta_n', 'm_bar_0',
        'c_uc', 'e', 'c_us', 'sqrt_a',
        't_0e', 'c_ic', 'omega_0', 'c_is',
        'i_0', 'c_rc', 'w', 'omega_dot',
        'i_dot', 'code_l2', 'gps_week', 'l2_p',
        'sv_acc', 'sv_health', 'tgd', 'iod_clock',
        'trans_time', 'fit_int', 'spare_1', 'spare_2'
    ]

    records = dict(zip(record_keys, record_fields))
    return epoch | records


def _parse_galileo_entry(lines: list[str]) -> dict[str, any]:

    epoch = _parse_epoch_fields(lines[0])
    record_fields = _parse_record_fields(lines)

    # Satellite System Records (see page 68)
    record_keys = [
        'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
        'iod_nav', 'c_rs', 'delta_n', 'm_bar_0',
        'c_uc', 'e', 'c_us', 'sqrt_a',
        't_0e', 'c_ic', 'omega_0', 'c_is',
        'i_0', 'c_rc', 'w', 'omega_dot',
        'i_dot', 'data_source', 'gal_week', 'spare_1',
        'sisa', 'sv_health', 'bgd_e5a_e1', 'bgd_e5b_e1',
        'trans_time', 'spare_2', 'spare_3', 'spare_4'
    ]

    records = dict(zip(record_keys, record_fields))
    return epoch | records


def _parse_gnss_entry(lines: list[str]) -> dict[str, any]:

    epoch = _parse_epoch_fields(lines[0])
    record_fields = _parse_record_fields(lines)

    # Satellite System Records (see page 68)
    record_keys = [
        'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
        'sat_pos_x', 'sat_vel_x', 'sat_acc_x', 'health',
        'sat_pos_y', 'sat_vel_y', 'sat_acc_y', 'freq_num',
        'sat_pos_z', 'sat_vel_z', 'sat_acc_z', 'age',
        'status', 'l1_l2_delay', 'urai', 'health_flags',
    ]

    records = dict(zip(record_keys, record_fields))

    to_return = epoch | records
    _add_missing_fields(to_return)
    return to_return


def _parse_qzss_entry(lines: list[str]) -> dict[str, any]:

    epoch = _parse_epoch_fields(lines[0])
    record_fields = _parse_record_fields(lines)

    # Satellite System Records (see page 68)
    record_keys = [
        'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
        'iod_eph', 'c_rs', 'delta_n', 'm_bar_0',
        'c_uc', 'e', 'c_us', 'sqrt_a',
        't_0e', 'c_ic', 'omega_0', 'c_is',
        'i_0', 'c_rc', 'w', 'omega_dot',
        'i_dot', 'code_l2', 'gps_week', 'l2p_flags',
        'sv_acc', 'sv_health', 'tgd', 'iod_clock',
        'trans_time', 'fit_int', 'spare_1', 'spare_2'
    ]

    records = dict(zip(record_keys, record_fields))
    return epoch | records


def _parse_bds_entry(lines: list[str]) -> dict[str, any]:

    epoch = _parse_epoch_fields(lines[0])
    record_fields = _parse_record_fields(lines)

    # Satellite System Records (see page 68)
    record_keys = [
        'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
        'aod_eph', 'c_rs', 'delta_n', 'm_bar_0',
        'c_uc', 'e', 'c_us', 'sqrt_a',
        't_0e', 'c_ic', 'omega_0', 'c_is',
        'i_0', 'c_rc', 'w', 'omega_dot',
        'i_dot', 'spare_1', 'bdt_week', 'spare_2',
        'sv_acc', 'sat_h1', 'tgd1_b1_b3', 'tgd2_b2_b3',
        'trans_time', 'aod_clock', 'spare_3', 'spare_4'
    ]

    records = dict(zip(record_keys, record_fields))
    return epoch | records


def _parse_sbas_entry(lines: list[str]) -> dict[str, any]:

    epoch = _parse_epoch_fields(lines[0])
    record_fields = _parse_record_fields(lines)

    # Satellite System Records (see page 68)
    record_keys = [
        'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
        'sat_pos_x', 'sat_vel_x', 'sat_acc_x', 'health',
        'sat_pos_y', 'sat_vel_y', 'sat_acc_y', 'acc_code',
        'sat_pos_z', 'sat_vel_z', 'sat_acc_z', 'iod_nav',
    ]

    records = dict(zip(record_keys, record_fields))

    to_return = epoch | records
    _add_missing_fields(to_return)
    return to_return


def _parse_irnss_entry(lines: list[str]) -> dict[str, any]:

    epoch = _parse_epoch_fields(lines[0])
    record_fields = _parse_record_fields(lines)

    # Satellite System Records (see page 68)
    record_keys = [
        'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
        'iod_eph_clock', 'c_rs', 'delta_n', 'm_bar_0',
        'c_uc', 'e', 'c_us', 'sqrt_a',
        't_0e', 'c_ic', 'omega_0', 'c_is',
        'i_0', 'c_rc', 'w', 'omega_dot',
        'i_dot', 'spare_1', 'irn_week', 'spare_2',
        'usr_rng_acc', 'health', 'tgd', 'spare_3',
        'trans_time', 'spare_4', 'spare_5', 'spare_6'
    ]

    records = dict(zip(record_keys, record_fields))
    return epoch | records


def _parse_record_fields(lines: list[str]) -> list[float]:

    fields: list[str] = []
    for line in lines:
        items = [line[4:23], line[23:42], line[42:61], line[61:80]]
        fields.extend(items)

    fields.pop(0)
    return [_str2float(f) for f in fields]


def _str2float(str_data: str) -> float:
    """
    Modified conversion for string to float values.
    Converts string in a Rinex file to float, adjusting for other
    exponent representations and replacing empty values with NaN.

    :param str_data: The exact string to be converts to float.
    :type str_data: str

    :return: The given string in the Rinex file converted to float.
    :rtype: float
    """
    str_value = str_data.replace("D", "E").strip()
    if str_value:
        return float(str_value)
    return float('nan')


def _add_missing_fields(records: dict[str, any]):

    # Satellite ECEF position in metres
    r_pz90f = np.array([
         records['sat_pos_x'],
         records['sat_pos_y'],
         records['sat_pos_z']
    ]) * 1e3

    # Satellite ECEF velocity in metres
    v_pz90f = np.array([
        records['sat_vel_x'],
        records['sat_vel_y'],
        records['sat_vel_z']
    ]) * 1e3

    # Convert extracted time fields to date time object
    time_fields = ['year', 'month', 'day', 'hour', 'minute', 'second']
    time_dict = {k: v for k, v in records.items() if k in time_fields}
    time_pz90f = datetime(**time_dict)

    # Convert given satellite position and velocity into ECI frame
    r_eci, v_eci = _pz90f2eci(time_pz90f, r_pz90f, v_pz90f)
    kep_params = _get_kepler_parameters(r_eci, v_eci)

    # Assign Kepler parameters
    records['sqrt_a'] = math.sqrt(kep_params['a'])
    records['e'] = kep_params['e']
    records['i_0'] = kep_params['i']
    records['omega_0'] = kep_params['Omega']
    records['w'] = kep_params['omega_small']
    records['m_bar_0'] = kep_params['theta']

    # Insert filler values for remaining fields
    records['t_0e'] = records['drift_rate_message_time']
    records['delta_n'] = 0
    records['i_dot'] = 0
    records['omega_dot'] = 0
    records['c_us'] = 0
    records['c_uc'] = 0
    records['c_is'] = 0
    records['c_ic'] = 0
    records['c_rs'] = 0
    records['c_rc'] = 0


def _pz90f2eci(time: datetime, r_pz90f: np.ndarray, v_pz90f: np.ndarray) -> tuple:
    """
    This function converts a position and velocity given in the PZ-90F
    coordinate reference frame used by the GLONASS system into the ECI
    reference frame.

    :param time: The datetime object for the time at which the PZ-90F
        parameters are correct
    :type time: datetime.datetime

    :param r_pz90f: A numpy array for the position in PZ-90F reference frame
    :type r_pz90f: numpy.ndarray (3-elements)

    :param v_pz90f: A numpy array for the velocity in PZ-90F reference frame
    :type v_pz90f: numpy.ndarray (3-elements)

    :returns:
        * rECI - numpy array for the position in the ECI reference frame
        * vECI - numpy array for the velocity in the ECI reference frame
    :rtype: tuple
    """

    theta_g0 = _greenwich_side_real_time_midnight(time)  # Greenwich Sidereal Time
    time_sec = ((time - datetime(time.year, time.month, time.day)).total_seconds())  # Seconds since midnight
    theta_ge = theta_g0 + OMEGA_E * time_sec  # Sidereal time at epoch

    # Convert position and velocity into ECI
    s_t = math.sin(theta_ge)
    c_t = math.cos(theta_ge)

    r_eci = np.array([
        r_pz90f[0] * c_t - r_pz90f[1] * s_t,
        r_pz90f[0] * s_t + r_pz90f[1] * c_t,
        r_pz90f[2]
    ])

    v_eci = np.array([
        v_pz90f[0] * c_t - v_pz90f[1] * s_t - OMEGA_E * r_eci[1],
        v_pz90f[0] * s_t + v_pz90f[1] * c_t + OMEGA_E * r_eci[0],
        v_pz90f[2]
    ])

    return r_eci, v_eci


def _greenwich_side_real_time_midnight(time: datetime) -> float:
    """
    This function calculates the Sidereal Time at Greenwich at 00:00:00 on
    the inputted day.

    :param time: A Python datetime object for the time in question
    :type time: datetime.datetime

    :return: The Greenwich Sidereal Time @ 00:00:00 of input day
    :rtype: float
    """
    time00 = datetime(time.year, time.month, time.day)  # 00:00:00 of day
    jd = _julian_date(time00)  # Julian date of midnight time
    ut = 0
    t = (jd - 2451545.0) / 36525.0
    t0 = 6.697374558 + (2400.051336 * t) + (0.000025862 * t ** 2) + (ut * 1.0027379093)
    gs_t0 = t0 % 24
    return gs_t0


def _julian_date(time: datetime) -> float:
    """
    This function gives the Julian Date of an input time, i.e. the time in days
    since 12:00 Jan 1 4713 B.C in Universal Time.

    :param time: A Python datetime object for the time in question
    :type time: datetime

    :returns: The JD - Julian date
    :rtype: float

    .. TODO:
        Currently only works for input times that are 00:00:00,
        this has a quick but currently unnecessary fix
    """

    year, month, day = int(time.year), int(time.month), int(time.day)

    dum1 = math.modf((month - 14.0) / 12.0)[1]
    jd = math.modf(1461 * (year + 4800 + dum1) / 4.0)[1]
    jd += math.modf((367 * (month - 2 - 12 * dum1)) / 12.0)[1]
    dum2 = math.modf((year + 4900 + dum1) / 100.0)[1]
    jd -= math.modf((3 * dum2) / 4.0)[1]
    jd += day - 2432075.5 + 2400000
    return jd


def _get_kepler_parameters(r_vec: np.ndarray, v_vec: np.ndarray) -> dict[str, float]:
    """
    This function converts a satellites position and velocity in the ECI
    frame into the keplerian parameters of the associated orbit as per the
    method shown in "Fundamentals of Astrodynamics and Applications",
    Vallado, 2007 (Algorithm 9, Page 113).

    :param r_vec: Satellite position in ECI frame (represented by r)
    :type r_vec: numpy.ndarray (3-elements)

    :param v_vec: Satellite velocity in ECI frame (represented by v)
    :type v_vec: numpy.ndarray (3-elements)

    :return: A dict containing all relevant Kepler orbital parameters
    :rtype: dict[str, float]
    """

    # Kepler value (μ)
    mu = GM

    # Calculate the angular momentum
    h_vec = np.cross(r_vec, v_vec, axis=0)
    h = norm(h_vec)

    # Calculate the node vector
    n_vec = np.cross(np.array([0, 0, 1]), h_vec, axis=0)
    n = norm(n_vec)

    # Calculate the l2 norms
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Calculate the Eccentricity vector
    e_vec = (v ** 2 - (mu / r)) * r_vec - (np.dot(r_vec, v_vec)) * v_vec
    e_vec /= mu
    e = norm(e_vec)

    # Specific mechanical energy (ξ)
    ke = ((v ** 2) / 2) - (mu / r)

    # Calculate semi major axis
    if e != 1.0:
        a = -mu / (2 * ke)
        p = a * (1 - e ** 2)
    else:
        a = np.inf
        p = h ** 2 / mu

    # TODO: Temp fix
    if (mu * a) < 0:
        return {'a': 0, 'e': np.nan, 'i': np.nan, 'Omega': np.nan,
                'omega_small': np.nan, 'theta': np.nan, 'tau': np.nan}

    # A correction previous put here?
    # cos_e = (1 - norm(r_vec) / a) / e
    # sin_e = r_vec.dot(v_vec) / (e * (mu * a) ** 0.5)
    # e = np.arctan2(sin_e, cos_e)

    # Calculate the inclination (i)
    i = np.arccos(h_vec[2] / h)

    # Calculate longitude of ascending node (Ω)
    # omega = np.arccos(n_vec[0] / n)
    omega = 0
    if n != 0:
        omega = _acos(n_vec[0] / n)
        if n_vec[1] < 0:
            omega = 2 * math.pi - omega

    # Argument of peri-centre (ω)
    # omega_small = np.arccos(n_vec.dot(e_vec) / (n * e))
    omega_small = 0
    if n != 0:
        omega_small = _acos(n_vec.dot(e_vec) / (n * e))
        if e_vec[2] < 0:
            omega_small = 2 * math.pi - omega_small

    # True anomaly at epoch time (v)
    # theta = np.arccos(e_vec.dot(r_vec) / (r * e))
    theta = 0
    if e != 0:
        theta = _acos(e_vec.dot(r_vec) / (r * e))
        if np.dot(r_vec, v_vec) < 0:
            theta = 2 * math.pi - theta

    # tau = -(e - e * math.sin(e)) / (mu * a ** -3) ** 0.5§

    return {
        'p': float(p),
        'a': float(a),
        'e': float(e),
        'i': float(i),
        'Omega': float(omega),
        'omega_small': float(omega_small),
        'theta': float(theta)
    }


def _acos(theta: float | np.ndarray) -> float:
    theta = max(-1.0, min(theta, 1.0))
    return math.acos(theta)


def _float(string: str) -> float or None:
    """
    This function is a minor amendment to the python "float" function,
    returning None for an input string of ``\\n`` rather than an error.

    :param string: The input string to be converted to a float
    :type string: str

    :return: A corresponding float value or None
    :rtype: float or None
    """
    if string.strip():
        return float(string)
    else:
        return None



if __name__ == '__main__':

    gps_save_dir = Path(__file__).parent.parent.parent / 'databases' / 'gps' / 'rinex'
    # assert gps_save_dir.is_dir()

    # gps_from_data = datetime(2019, 3, 1)
    # gps_to_data = datetime(2024, 11, 11)

    gps_from_data = datetime(2018, 1, 1)
    gps_to_data = datetime(2019, 3, 1)

    download_rinex_files(gps_from_data, gps_to_data, gps_save_dir, True)


    # gps_test_date = datetime(2024, 11, 1)
    # download_rinex_file(gps_test_date, gps_save_dir)

    # gps_save_dir.is_dir()


    # var = [s for s in SatelliteType]
    # print(var)

# if __name__ == '__main__':

    # import tempfile
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #
    #     test_dir = Path(tmp_dir)
    #     # test_date = datetime(2020, 11, 10)
    #     # test_date = datetime(2020, 9, 14)
    #     test_date = datetime(2024, 10, 7)
    #
    #     download_ephemeris_file(test_date, test_dir)
    #     test_file = get_rinex_path(test_date, test_dir)
    #     test_data = read_rinex_file(test_file)
    #
    # # test_file = Path(".") / '2024_278.rnx'
    # # test_data = read_ephemeris_file_new(test_file)
    # # test_result = ephemeris_to_satellite_params(test_data)

