
import pytest
import numpy as np

from pathlib import Path
from os.path import getsize
from qnav.output import formats
from abc import ABC, abstractmethod
from qnav.output.data import ResultsTable
from scipy.io import loadmat

def _assert_file_is_created(file_path: Path):
    """
    Asserts that a file has been created and written to.
    """
    assert file_path.is_file() and getsize(file_path) > 0, \
        'Output file was not successfully generated'

def _skip_if_missing(file_path: Path):
    """
    Used to skip tests if a required generated file is missing.
    """
    if not file_path.exists() or getsize(file_path) == 0:
        pytest.skip('Requires writing test to be executed first')

class TestFileFormat(ABC):

    @pytest.fixture(params=[1000], scope='module')
    def data_to_save(self, request) -> dict[str, np.ndarray]:
        """
        Returns test time steps and data records.
        """
        num_records: int = request.param
        return {
            'time_steps': np.arange(num_records),
            'position': np.random.rand(num_records, 3),
            'velocity': np.random.rand(num_records, 3),
            'acceleration': np.random.rand(num_records, 3),
            'attitude': np.random.rand(num_records, 3),
            'angle_rates': np.random.rand(num_records, 3),
        }

    @pytest.fixture(scope='module')
    def working_dir(self, tmp_path_factory) -> Path:
        """
        Returns a temporary directory to save the test results.
        """
        return tmp_path_factory.mktemp("format_testing")

    @abstractmethod
    @pytest.fixture(scope='module')
    def export_file(self, working_dir) -> Path:
        """
        Returns a temporary file to write results for testing to.
        """
        pass

    @pytest.fixture(scope='module')
    def results_table(self, data_to_save, working_dir) -> ResultsTable:
        """
        Return already written results.
        """
        results_file = working_dir / 'results.qnr'
        rt = ResultsTable(results_file)
        rt.write('Testing results', **data_to_save)
        return rt

    @pytest.fixture(scope='module')
    def expected(self, results_table: ResultsTable) -> np.ndarray:
        """
        Returns excepted data from generated results.
        """
        data = results_table.get_data()
        return np.array(data['table']['values'])

    @abstractmethod
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        """
        Verifies that results are written correctly.
        """
        pass

    @abstractmethod
    def test_reading(self, expected: np.ndarray, export_file: Path):
        """
        Verifies that results are read correctly.
        """
        pass


class TestCSV(TestFileFormat):
    """
    Executes tests for the CSV file format.
    """

    @pytest.fixture(scope='class')
    def export_file(self, working_dir) -> Path:
        return working_dir / 'results.csv'

    @pytest.mark.order(1)
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        formats.save_as_csv(export_file, results_table)
        _assert_file_is_created(export_file)

    @pytest.mark.order(2)
    def test_reading(self, expected: np.ndarray, export_file: Path):
        _skip_if_missing(export_file)
        data = np.loadtxt(export_file, delimiter=',')
        np.testing.assert_array_equal(expected, data, 'data not matching')


class TestCompCSV(TestCSV):
    """
    Executes tests for the compressed CSV file format.
    """

    @pytest.fixture(scope='class')
    def export_file(self, working_dir) -> Path:
        return working_dir / 'results.csv.gz'

    @pytest.mark.order(1)
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        formats.save_as_csv_comp(export_file, results_table)
        assert export_file.exists()


class TestNumpy(TestFileFormat):
    """
    Executes tests for the numpy array file format.
    """

    @pytest.fixture(scope='class')
    def export_file(self, working_dir) -> Path:
        return working_dir / 'results.npy'

    @pytest.mark.order(1)
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        formats.save_as_numpy(export_file, results_table)
        _assert_file_is_created(export_file)

    @pytest.mark.order(2)
    def test_reading(self, expected: np.ndarray, export_file: Path):
        _skip_if_missing(export_file)
        data = np.load(export_file, allow_pickle=False)
        np.testing.assert_array_equal(expected, data, 'data not matching')


class TestCompNumpy(TestFileFormat):
    """
    Executes tests for the compressed numpy array file format.
    """

    @pytest.fixture(scope='class')
    def export_file(self, working_dir) -> Path:
        return working_dir / 'results.npz'

    @pytest.mark.order(1)
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        formats.save_as_numpy_comp(export_file, results_table)
        _assert_file_is_created(export_file)

    @pytest.mark.order(2)
    def test_reading(self, expected: np.ndarray, export_file: Path):
        _skip_if_missing(export_file)
        load_data = np.load(export_file, allow_pickle=False)
        data = load_data[load_data.files[0]]
        np.testing.assert_array_equal(expected, data, 'data not matching')


class TestBinary(TestFileFormat):
    """
    Executes tests for the binary file format.
    """

    @pytest.fixture(scope='class')
    def export_file(self, working_dir) -> Path:
        return working_dir / 'results.bin'

    @pytest.mark.order(1)
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        formats.save_as_binary(export_file, results_table)
        _assert_file_is_created(export_file)

    @pytest.mark.order(2)
    def test_reading(self, expected: np.ndarray, export_file: Path):
        _skip_if_missing(export_file)
        data = np.fromfile(export_file, dtype=float)
        np.testing.assert_array_equal(
            expected.ravel(), data,
            'data not matching')


class TestMatlab(TestFileFormat):
    """
    Executes tests for the MATLAB save file format.
    """

    @pytest.fixture(scope='class')
    def export_file(self, working_dir) -> Path:
        return working_dir / 'results.mat'

    @pytest.mark.order(1)
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        formats.save_as_matlab(export_file, results_table)
        _assert_file_is_created(export_file)

    @pytest.mark.order(2)
    def test_reading(self, expected: np.ndarray, export_file: Path):
        _skip_if_missing(export_file)
        load_data = loadmat(str(export_file))
        data = load_data['values']
        np.testing.assert_array_equal(expected, data, 'data not matching')


class TestCompMatlab(TestMatlab):
    """
    Executes tests for the compressed MATLAB save file format.
    """

    @pytest.fixture(scope='class')
    def export_file(self, working_dir) -> Path:
        return working_dir / 'results.z.mat'

    @pytest.mark.order(1)
    def test_writing(self, results_table: ResultsTable, export_file: Path):
        formats.save_as_matlab_comp(export_file, results_table)
        _assert_file_is_created(export_file)
