

import numpy as np

from dataclasses import dataclass
from scipy.io import savemat
from pathlib import Path
from enum import Enum


from qnav.output.data import ResultsTable


@dataclass
class FileFormat:
    """
    Defines the fields for each supported file format.
    Included as a simpler way to represent the SaveFormat enum class.
    """
    match_id: str       # A name used for identifying the file format.
    file_ext: str       # The file extension used by the file format.


class SaveFormat(FileFormat, Enum):
    """
    A collection of the supported file formats for results table.
    Controllable flags for selecting the use of the supported file formats
    when exporting data. A useful way to efficiently export data in
    multiple formats, without employing multiple functions.
    """

    CSV = 'csv', '.csv'                     # Comma Seperated Value format (.csv)
    CSV_COMP = 'csv_comp', '.csv.gz'        # Comma Seperated Value compress (.csv.gz)
    NUMPY = 'numpy', '.npy'                 # Numpy array file format (.np)
    NUMPY_COMP = 'numpy_comp', '.npz'       # Numpy array compressed file format (.npz)
    BINARY = 'binary', '.bin'               # Standard 64-bit float binary format (.bin)
    MATLAB = 'matlab', '.mat'               # MATLAB save file format (.mat)
    MATLAB_COMP = 'matlab_comp', '.z.mat'   # MATLAB save file format compressed (.z.mat)

    def __str__(self) -> str:
        """
        Displays the Enum instance as a clean string.

        :return: Simply returns the name of the enum for strings.
        :rtype: str
        """
        return self.name



def __get_data_values(results: ResultsTable, down_sample: int = 1) -> np.ndarray:
    data = results.get_data()
    values = data['table']['values']
    return values[::down_sample, :]


def __get_data_columns(results: ResultsTable) -> np.ndarray:
    data = results.get_data()
    return data['table']['columns']

def save_as_csv(file_path: Path, results: ResultsTable, down_sample: int = 1):
    """
    Exports the given results table as a CSV file.

    :param file_path: The file path to save the CSV data to.
    :type file_path: pathlib.Path

    :param results: The results to convert to CSV.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int
    """
    columns = __get_data_columns(results)
    values = __get_data_values(results, down_sample)
    header_text = ", ".join(columns)
    np.savetxt(file_path, values, delimiter=",", header=header_text)

def save_as_csv_comp(file_path: Path, results: ResultsTable, down_sample: int = 1):
    """
    Exports the given results table as a compressed CSV file.

    :param file_path: The file path to save the compressed CSV data to.
    :type file_path: pathlib.Path

    :param results: The results to convert to compressed CSV.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int
    """
    save_as_csv(file_path.with_suffix('.gz'), results, down_sample)

def save_as_numpy(file_path: Path, results: ResultsTable, down_sample: int = 1):
    """
    Exports the given results table as a numpy data file.

    :param file_path: The file path to save the numpy data to.
    :type file_path: pathlib.Path

    :param results: The results to convert to numpy save file.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int
    """
    values = __get_data_values(results, down_sample)
    np.save(file_path, values, False)

def save_as_numpy_comp(file_path: Path, results: ResultsTable, down_sample: int = 1):
    """
    Exports the given results table as a compressed numpy data file.

    :param file_path: The file path to save the compressed numpy data to.
    :type file_path: pathlib.Path

    :param results: The results to convert to compressed numpy save file.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int
    """
    values = __get_data_values(results, down_sample)
    np.savez_compressed(file_path, values)

def save_as_binary( file_path: Path, results: ResultsTable, down_sample: int = 1):
    """
    Exports the given results table as a compressed numpy data file.

    :param file_path: The file path to save the compressed numpy data to.
    :type file_path: pathlib.Path

    :param results: The results to convert to compressed numpy save file.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int
    """
    values = __get_data_values(results, down_sample)
    values.tofile(file_path, format='%.18e')

def __save_as_mat(file_path: Path, results: ResultsTable, do_compression: bool, down_sample: int = 1):
    data = results.get_data()
    values = data['table']['values'][::down_sample, :]
    columns = data['table']['columns']
    to_save = {'columns': columns, 'values': values}
    savemat(file_path, to_save, format='5', do_compression=do_compression)

def save_as_matlab(file_path: Path, results: ResultsTable, down_sample: int = 1):
    """
    Exports the given results table as a (non-compressed) MATLAB data file.

    :param file_path: The file path to save the MATLAB data to.
    :type file_path: pathlib.Path

    :param results: The results to convert to MATLAB file.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int
    """
    __save_as_mat(file_path, results, False, down_sample)

def save_as_matlab_comp(file_path: Path, results: ResultsTable, down_sample: int = 1):
    """
    Exports the given results table as a MATLAB data file (with compression).

    :param file_path: The file path to save the MATLAB data to.
    :type file_path: pathlib.Path

    :param results: The results to convert to MATLAB file.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int
    """
    __save_as_mat(file_path, results, True, down_sample)

def save_as(file_dir: Path, file_name: str, results: ResultsTable,
            down_sample: int = 1, *formats: SaveFormat):
    """
    Saves the given results as one or format types.

    :param file_dir: The location to write results to.
    :type file_dir: pathlib.Path

    :param file_name: The filename prefix to use.
    :type file_name: str

    :param results: The results to convert and write.
    :type results: ResultsTable

    :param down_sample: Optional down-sampling factor to apply.
    :type down_sample: int

    :param formats: The output formats to use.
    :type formats: SaveFormat
    """

    file_path = file_dir / file_name

    if not results.is_data_loaded:
        return

    if not file_dir.exists():
        file_dir.mkdir(parents=True)

    for sf in SaveFormat:
        if sf in formats:

            save_file = file_path.with_suffix(sf.file_ext)

            match sf:

                case SaveFormat.CSV:
                    save_as_csv(save_file, results, down_sample)

                case SaveFormat.CSV_COMP:
                    save_as_csv_comp(save_file, results, down_sample)

                case SaveFormat.NUMPY:
                    save_as_numpy(save_file, results, down_sample)

                case SaveFormat.NUMPY_COMP:
                    save_as_numpy_comp(save_file, results, down_sample)

                case SaveFormat.BINARY:
                    save_as_binary(save_file, results, down_sample)

                case SaveFormat.MATLAB:
                    save_as_matlab(save_file, results, down_sample)

                case SaveFormat.MATLAB_COMP:
                    save_as_matlab_comp(save_file, results, down_sample)


# if __name__ == '__main__':
#
#     test_format = SaveFormat.MATLAB_COMP
#     print(test_format)

    # test_file = Path("testing.qnr")
    # test_rt = ResultsTable(test_file, auto_load=True)
    # # test_data = test_rt.get_data()
    #
    # output_dir = Path()

    # save_as_csv(output_dir / "testing.csv", test_rt)
    # save_as_csv_comp(output_dir / "testing.csv.gz", test_rt)
    # save_as_numpy(output_dir / "testing.npy", test_rt)
    # save_as_numpy_comp(output_dir / "testing.npz", test_rt)
    # save_as_matlab(output_dir / "testing.mat", test_rt)
    # save_as_matlab_comp(output_dir / "testing.z.mat", test_rt)
