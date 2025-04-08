
import pytest

from pathlib import Path

from qnav.input.config_handler import ConfigHandler


# All section names used in the testsing configuration file
_ALL_SECTIONS = ['Strings', 'Integers', 'Floats', 'Exponents',
                 'Boolean', 'Paths', 'Interpolated', 'SubPath']

class TestConfig:
    """
    Tests cases used for testing the reading of a configuration file.
    """

    @staticmethod
    @pytest.fixture(scope='class')
    def config_file() -> Path:
        """
        Returns the path to the test configuration file.
        """
        file_name = 'example_config.ini'
        return Path(__file__).parent / file_name

    @staticmethod
    @pytest.fixture(scope='function')
    def config_handler(config_file: Path) -> ConfigHandler:
        """
        Returns a ConfigHandler instance for the configuration file.
        """
        assert config_file.is_file(), "Missing test configuration file"
        return ConfigHandler(config_file)

    @staticmethod
    def test_config_name(config_handler: ConfigHandler, config_file: Path):
        expected = config_file.name
        assert config_handler.get_config_name() == expected

    @staticmethod
    def test_section_names(config_handler: ConfigHandler):
        expected = _ALL_SECTIONS
        returned = config_handler.get_sections()
        assert set(returned) == set(expected)

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('stringValue1', 'String'),
        ('stringValue2', 'A Longer String'),
        ('stringValue3', '12345678910')
    ])
    def test_reading_strings(config_handler: ConfigHandler, key_values: tuple[str, str]):
        key, expected = key_values
        read_value = config_handler.get_str('Strings', key)
        assert read_value == expected, "String values do not match"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('intValue1', 123),
        ('intValue2', -56),
        ('intValue3', 78),
    ])
    def test_reading_integers(config_handler: ConfigHandler, key_values: tuple[str, int]):
        key, expected = key_values
        read_value = config_handler.get_int('Integers', key)
        assert read_value == expected, "Integer values do not match"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('floatValue1', 1.234567),
        ('floatValue2', 0.0000001379),
        ('floatValue3', -1234.6789),
    ])
    def test_reading_floats(config_handler: ConfigHandler, key_values: tuple[str, float]):
        key, expected = key_values
        read_value = config_handler.get_float('Floats', key)
        assert read_value == expected, "Float values do not match"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('exponentValue1', 1e6),
        ('exponentValue2', 1e-6),
        ('exponentValue3', -5e-6),
    ])
    def test_reading_exponents(config_handler: ConfigHandler, key_values: tuple[str, float]):
        key, expected = key_values
        read_value = config_handler.get_float('Exponents', key)
        assert read_value == expected, "Exponent values do not match"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('trueBool', True),
        ('falseBool', False),
        ('onBool', True),
        ('offBool', False),
        ('yesBool', True),
        ('noBool', False),
        ('zeroBool', False),
        ('oneBool', True)
    ])
    def test_reading_booleans(config_handler: ConfigHandler, key_values: tuple[str, bool]):
        key, expected = key_values
        read_value = config_handler.get_bool('Boolean', key)
        assert read_value == expected, "Boolean values do not match"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('posixPath', Path('posix/parent_dir/file.suffix')),
        ('windowsPath', Path('windows/parent_dir/file.suffix')),
    ])
    def test_reading_paths(config_handler: ConfigHandler, key_values: tuple[str, Path]):
        key, expected = key_values
        read_value = config_handler.get_value_filepath(
            'Paths', key, None, False)
        assert expected.resolve() == read_value.resolve(), "Path values do not match"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('interpValues1', 'String'),
        ('interpValues2', 'String 1.234567'),
    ])
    def test_reading_interpolated(config_handler: ConfigHandler, key_values: tuple[str, str]):
        key, expected = key_values
        read_value = config_handler.get_str('Interpolated', key)
        assert read_value == expected, "Interpolated values do not match"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('subConfig', 'example_subconfig.ini', 'externalString1', 'I\'m external'),
        ('subConfig', 'example_subconfig.ini', 'externalString2', '1234.5678'),
    ])
    def test_reading_sub_files(config_handler: ConfigHandler, key_values: tuple[str, str, str, str]):

        section = 'SubPath'
        key, path, sub_key, expected = key_values
        read_path = config_handler.get_str(section, key)
        assert path == read_path, "Unexpected path read"

        to_append = str(Path(__file__).parent / read_path)
        config_handler.append_config(to_append, section, section)

        read_value = config_handler.get_str(section, sub_key)
        assert read_value == expected, "Sub-configuration values do not match"


    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('DoesNotExist', 'Fallback'),
    ])
    def test_fallback_strings(config_handler: ConfigHandler, key_values: tuple[str, str]):
        key, fallback = key_values
        read_value = config_handler.get_str('Strings', key, fallback)
        assert read_value == fallback, "Fallback string not returned"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('DoesNotExist', 9876),
    ])
    def test_fallback_ints(config_handler: ConfigHandler, key_values: tuple[str, int]):
        key, fallback = key_values
        read_value = config_handler.get_int('Integers', key, fallback)
        assert read_value == fallback, "Fallback integer not returned"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('DoesNotExist', 0.12345),
    ])
    def test_fallback_floats(config_handler: ConfigHandler, key_values: tuple[str, float]):
        key, fallback = key_values
        read_value = config_handler.get_float('Floats', key, fallback)
        assert read_value == fallback, "Fallback float not returned"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('DoesNotExist', 1e-9),
    ])
    def test_fallback_exponent(config_handler: ConfigHandler, key_values: tuple[str, float]):
        key, fallback = key_values
        read_value = config_handler.get_float('Exponents', key, fallback)
        assert read_value == fallback, "Fallback exponent not returned"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('DoesNotExist', True),
        ('DoesNotExist', False),
    ])
    def test_fallback_boolean(config_handler: ConfigHandler, key_values: tuple[str, bool]):
        key, fallback = key_values
        read_value = config_handler.get_bool('Boolean', key, fallback)
        assert read_value == fallback, "Fallback boolean not returned"

    @staticmethod
    @pytest.mark.parametrize('key_values', [
        ('DoesNotExist', Path("a/b/c/d.txt")),
    ])
    def test_fallback_paths(config_handler: ConfigHandler, key_values: tuple[str, Path]):
        key, fallback = key_values
        read_value = config_handler.get_value_filepath('Boolean', key, fallback)
        assert read_value == fallback, "Fallback path not returned"