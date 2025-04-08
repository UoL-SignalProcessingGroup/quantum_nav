
import os
import pytest
import numpy as np

from qnav.output.data import ResultsTable



def _assert_desc_match(data: dict[str, any], expected: str):
    assert data['description'] == expected, \
        'Description mismatch'

def _assert_time_match(data: dict[str, any], expected: dict[str, np.ndarray]):
    np.testing.assert_array_equal(
        data['time_steps'], expected['time_steps'],
        'Time steps mismatch')

def _assert_fields_match(data: dict[str, any], expected: dict[str, np.ndarray]):
    for key, value in expected.items():
        if key != 'time_steps':
            np.testing.assert_array_equal(
                data['data'][key], expected[key],
                f'Field mismatch for "{key}"')

def _assert_is_mem_map(data: dict[str, any]):
    for key, value in data['data'].items():
        assert isinstance(value, np.memmap), \
            f'Field "{key}" is not a memmap instance'

    assert isinstance(data['time_steps'], np.memmap), \
            'Read time steps are not a memmap instance'

def _assert_is_not_mem_map(data: dict[str, any]):
    for key, value in data['data'].items():
        assert not isinstance(value, np.memmap), \
            f'Field "{key}" is not a memmap instance'

    assert not isinstance(data['time_steps'], np.memmap), \
            'Read time steps are not a memmap instance'


class TestResults:

    @pytest.fixture(params=[1, 100, 10000], scope='class')
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

    @pytest.fixture(scope='function')
    def results_dir(self, tmp_path_factory):
        """
        Returns a temporary directory to save the test results.
        """
        return tmp_path_factory.mktemp("results_testing")

    @staticmethod
    def test_writing(data_to_save, results_dir):
        results_file = results_dir / 'results.qnr'
        rt = ResultsTable(results_file)

        rt.write('Testing results', **data_to_save)
        assert results_file.is_file(), 'Could not find results file'
        assert os.path.getsize(results_file) > 0, 'Results file is empty'

    @staticmethod
    def test_reading(data_to_save, results_dir):

        # Obtain saving information
        desc = 'Testing results'
        results_file = results_dir / 'results.qnr'

        # Generate and write the results data
        rt = ResultsTable(results_file)
        rt.write(desc, **data_to_save)

        # Read back in the results data using another table.
        other_rt = ResultsTable(results_file, auto_load=True)
        read_data = other_rt.get_data()

        # Assert that all read values match those written
        _assert_desc_match(read_data, desc)
        _assert_time_match(read_data, data_to_save)
        _assert_fields_match(read_data, data_to_save)

    @staticmethod
    def test_is_loaded(data_to_save, results_dir):

        results_file = results_dir / 'results.qnr'
        rt = ResultsTable(results_file)

        # Check before and after writing data
        assert not rt.is_data_loaded, 'States that data is loaded'
        rt.write('Testing results', **data_to_save)
        assert rt.is_data_loaded, 'States that data is not loaded'

    @staticmethod
    @pytest.mark.parametrize('do_load', [True, False])
    def test_auto_load(data_to_save, results_dir, do_load):

        results_file = results_dir / 'results.qnr'
        rt = ResultsTable(results_file)
        rt.write('Description', **data_to_save)

        other_rt = ResultsTable(results_file, auto_load=do_load)
        assert do_load == other_rt.is_data_loaded, 'Incorrect is_data_loaded value'

        try:
            other_rt.get_data()
        except LookupError:
            assert not do_load, \
                'Exception encountered when not expected'

        other_rt.read()
        assert rt.is_data_loaded, 'Data should be loaded by now'


    @staticmethod
    @pytest.mark.parametrize('read_only', [True, False])
    def test_read_only(data_to_save, results_dir, read_only):

        results_file = results_dir / 'results.qnr'
        rt = ResultsTable(results_file)
        desc = 'Description'
        rt.write(desc, **data_to_save)

        new_desc = "New description"
        new_data = {k: v*2 for k, v in data_to_save.items()}
        other_rt = ResultsTable(results_file, read_only=read_only)

        try:
            other_rt.write(new_desc, **new_data)
        except ValueError:
            assert read_only, \
                'Exception encountered when not expected'

        # TODO: Also test attempts to change values?

        another_rt = ResultsTable(results_file, auto_load=True)
        read_data = another_rt.get_data()

        desc = desc if read_only else new_desc
        expect = data_to_save if read_only else new_data

        _assert_desc_match(read_data, desc)
        _assert_time_match(read_data, expect)
        _assert_fields_match(read_data, expect)

    @staticmethod
    @pytest.mark.parametrize('use_virtual', [True, False])
    def test_virtual_memory(data_to_save, results_dir, use_virtual):

        # Generate and write the results data
        results_file = results_dir / 'results.qnr'
        rt = ResultsTable(results_file)
        rt.write('Testing results', **data_to_save)

        # Read results data using virtual memory option
        other_rt = ResultsTable(results_file, auto_load=False)
        other_rt.read(virtual_memory=use_virtual)
        read_data = other_rt.get_data()

        if use_virtual:
            _assert_is_mem_map(read_data)
        else:
            _assert_is_not_mem_map(read_data)
