"""
=================
config_handler.py
=================

:summary:
    This class provides a wrapper for Python's ConfigHandler class.
    This supplies additional functionality such as automatically reading from
    sub-configuration files and providing platform independent means for
    extracting configuration values in usable formats.

:authors:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of class.
"""

from configparser import ExtendedInterpolation
from configparser import NoSectionError
from configparser import NoOptionError
from configparser import ConfigParser
from pathlib import PureWindowsPath
from pathlib import Path

from numpy import typing as npt
from shutil import copyfile
from warnings import warn

import numpy as np
import re

# The name of the default settings configuration file.
_DEFAULT_CONFIG: Path = Path("default_settings.ini")


class ConfigHandler:
    """
    This class provides means for reading the user's configuration preferences
    from a given configuration file. Once read, values can be extracted within
    various formats (strings, booleans, floats, arrays, etc..). This class is
    essentially a wrapper for Python's ConfigHandler class, providing
    additional functionality such as the ability to split configuration
    across multiple files.
    """

    # The ConfigParser used for reading the given config file.
    parser = None

    # The default configuration file to use.
    config_file: Path = None

    # Whether a warning should be displayed if a default value is used.
    warn_on_fallback: bool = True

    def __init__(self, config_file: Path = _DEFAULT_CONFIG):
        """
        The constructor for this class. Reads the settings from  a given config
        file. If no config file is supplied, a specified default file will be
        used instead. Once initialised, values can be extracted from the
        configuration file during runtime in requested formats.

        :param config_file: (Optional) The location of the configuration file
            to read from. If not provided, the default configuration file
            will be used automatically (default=``_DEFAULT_CONFIG``).
        :type config_file: str
        """

        # Load the given configuration file (or the default file).
        self.config_file = config_file
        self.load_configuration()

    def get_config_name(self) -> str:
        """
        This simple function returns the name of the configuration file being
        used. This function will automatically remove the project database_dir
        from the returned name (if the selected config is located within the
        project database_dir).

        :return: The short name of the selected configuration file that is
            being used.
        :rtype: str
        """

        # Extract the current configuration file
        # and the project's database_dir name.
        # config_file_str = self.config_file.name
        # parent_dir_str = str(PROJECT_PATH / "a")[:-1]
        #
        # # Finally, return the short name of the config file.
        # return config_file_str.replace(parent_dir_str, "")
        return self.config_file.name

    def load_configuration(self):
        """
        This function reads and records the content of the given configuration
        file (and expected sub-files). Expected sections/sensors are
        automatically searched and their sub-files are also included within
        the loading process. This function can be used to re-load/re-read the
        information from configuration files (if changes are made during
        runtime).
        """

        try:
            # Try to read from the configuration file using extended interpolation.
            # This allows variables to be used within the config file.
            self.parser = ConfigParser(interpolation=ExtendedInterpolation())
            self.parser.read(self.config_file)

            # If the file is empty or could not be read, throw an exception.
            if len(self.parser.sections()) == 0:
                raise EOFError(f"No settings found in file '{self.config_file}'!")

        except IOError:

            # If the file could not be read from, throw an exception.
            raise IOError(f"File '{self.config_file}' could not be read!")


    def append_config(self, config_file: str, from_section: str, to_section: str, to_overwrite: bool = False):
        """
        This function will read and record the values of an external
        configuration files (i.e. a sub-configuration) and merge the read
        information with the current ConfigHandler instance.

        :param config_file: The sub-configuration file to read.
        :type config_file: str

        :param from_section: The section within the sub-configuration
            file to read and record.
        :type from_section: str

        :param to_section: The section within the current instance to append
            the read information to (can be a new section).
        :type to_section: str

        :param to_overwrite: If set to true, previous existing values from
            the main configuration will be overwritten by those read from
            sub-configuration files (default=False).
        :type to_overwrite: bool
        """

        # Identify the sub-configuration file.
        sub_config = Path(config_file)

        # Ensure the sub-file exists before continuing:
        if not sub_config.is_file():

            # If not, raise an error.
            raise FileNotFoundError(
                f"The sub-file '{sub_config}' does not exist!")

        # Attempt to parse the given sub-configuration file.
        parser = ConfigParser(interpolation=ExtendedInterpolation())
        parser.read(sub_config)

        # Ensure that the sub-configuration file can be read.
        if len(parser.sections()) == 0:
            raise IOError(f"No settings could be read from sub-file '{sub_config}'!")

        # If the given new section has not been created before, create it.
        if not self.parser.has_section(to_section):
            self.parser.add_section(to_section)

        # Append each of the items read from the sub-configuration file to the
        # current ConfigHandler instance under the new section. Update
        # previously existing values depending on the to_overwrite option.
        for key, value in parser.items(from_section):
            if not (self.parser.has_option(to_section, key) and to_overwrite):
                self.parser.set(to_section, key, value)


    def expand_section(self, section: str, path_property: str,
                       sub_file_section: str = None, to_overwrite: bool = False):
        """
        Expand section by importing the referenced sub-configuration path.
        Imports additional properties into section, for a given section and
        property referencing the sub-configuration file.

        :param section: The section containing the property with the
            sub-configuration path.
        :type section: str

        :param path_property: The property containing the sub-
            configuration path.
        :type path_property: str

        :param sub_file_section: (Optional) The section to import from the
            sub-configuration file.
        :type sub_file_section: str

        :param to_overwrite: Should this import overwrite existing settings?
        :type to_overwrite: bool
        """

        # Set optional arguments:
        if sub_file_section is None:
            sub_file_section = section

        # If a value is set for the given property:
        if self.is_value_set(section, path_property):

            # Get the file given as the property value:
            sub_file = self.get_value_filepath(
                section, path_property, must_exist=True)

            # Attempt to append the sub configuration
            self.append_config(
                    str(sub_file), sub_file_section,
                    section, to_overwrite)


    def quick_append_config(self,
                            from_section: npt.ArrayLike,
                            from_variable: npt.ArrayLike,
                            external_section: npt.ArrayLike,
                            to_overwrite: bool = False):
        """
        This function simplifies merging multiple configuration file sections
        to the current configuration.

        :param from_section: The sections within the main configuration file
            that contain the variables to read from.
        :type from_section: numpy.typing.ArrayLike

        :param from_variable: The filepath variable names from the given
            section stating the external configuration files to read from.
        :type from_variable: numpy.typing.ArrayLike

        :param external_section: The external sub-configuration file sections
            to merge with the main configuration.
        :type external_section: numpy.typing.ArrayLike

        :param to_overwrite: If set to true, previous existing values from
            the main configuration will be overwritten by those read from
            sub-configuration files (default=False).
        :type to_overwrite: bool
        """

        # Ensure the given input is of equal size.
        assert len(from_section) == len(from_variable) == len(external_section), \
            "The given input lists must be of the same size!"

        # For each element within the given lists:
        for section, value, ext in zip(from_section, from_variable, external_section):

            # If the current section and value exist:
            if self.parser.has_option(section, value):

                # Extract the current value.
                config_file = self.get_str(section, value)

                try:
                    # Attempt to append to the current configuration.
                    self.append_config(config_file, ext, section, to_overwrite)

                except Exception as e:
                    # In the event of a reading error, return a custom error.
                    e_type = e.__class__.__name__
                    raise NavConfigError(section, value, e_type, str(e))

    def get_all_values(self) -> np.ndarray:
        """
        This function returns a matrix containing all the values read from the
        given configuration file(s). This function is namely for debugging
        purposes. Values are displayed as raw strings.

        :return: An N-by-3 matrix of values containing the section, variable
            name and value for each option read.
        :rtype: numpy.ndarray
        """

        # The final matrix to return.
        to_return = None

        # For each of the read sections:
        for section in self.get_sections():

            # Obtain a matrix of values for the current section.
            values = np.array(self.parser.items(section))

            # Create an equal sized column repeating the current section name,
            # then append it to the above matrix.
            size = values.shape[0]
            column = np.repeat(section, size).reshape((size, 1))
            to_append = np.concatenate((column, values), axis=1)

            # Append the resulting matrix to the return matrix.
            if to_return is None:
                to_return = to_append
            else:
                to_return = np.concatenate((to_return, to_append), axis=0)

        # Remove comments from the returned values to prevent confusion.
        for i in range(to_return.shape[0]):
            to_return[i, 2] = str(to_return[i, 2]).split("#", 1)[0].strip()
            # to_return[i, 2] = to_return[i, 2].split("#", 1)[0].strip()

        # Finally, return the resulting matrix.
        return to_return

    def get_sections(self):
        """
        This function returns a list of all the sections read from the given
        configuration file(s). This is useful for debugging and ensure
        required content is being correctly read.

        :return: A list containing all section names.
        :rtype: list
        """
        return self.parser.sections()

    def _missing_value_warning(self, section: str, name: str, fallback_value):
        """
        If requested, this function prints a warning each time a requested
        parameter could not be found from the provided configuration file.
        This is useful for debugging and finding non-supplied variables that
        have been overridden with hardcoded defaults.

        :param section: The section name where the missing value/parameter was
            to be found under.
        :type section: str

        :param name: The name of the missing value/parameter.
        :type name: str

        :param fallback_value: The hardcoded default value that has been
            automatically use inplace of the missing value.
        :type fallback_value: any
        """

        # If an actual default value is being used with warnings enabled:
        if self.warn_on_fallback and fallback_value is not None:

            # Ensure the used default value is human-readable.
            fallback_str = fallback_value if fallback_value is not None else "None"

            # Construct a warning message to be presented to the user.
            warning_msg = f'\n\tSection:\t\t{section}\n\tVariable:\t{name}' \
                          f'\n\n\tWarning:\t Missing value automatically' \
                          f' set to \'{fallback_str}\''

            # Produce the above warning.
            warn(warning_msg, NavConfigWarning)

    def _missing_value_error(self, section: str, name: str):
        """
        This function throws an error for when a mandatory parameter could
        not be found from the provided configuration file. Unlike missing
        value warnings, this error cannot be suppressed or ignored by the
        user.

        :param section: The section name where the missing value/parameter was
            to be found under.
        :type section: str

        :param name: The name of the missing value/parameter.
        :type name: str

        :raises: KeyError
        """

        raise NavConfigError(
            section, name, "MissingValue",
            f"Missing entry for required variable!")

    def _get_value(self, section: str, name: str, fallback=None):
        """
        Attempts to return the corresponding named value under the given
        section. If the requested value could not be found, an error will be
        thrown unless a suitable default/default value has been given. In
        the case of the latter the default will be used, but a warning may
        be displayed to the user (if enabled).

        This function returns values as the (string) text parsed from the
        configuration file. Therefore, for additional data formats other
        functions must be supplied that extend from this one.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: any

        :return: Returns either the string found within the config file
            for the given section and name, or the default value if not
            found.
        :rtype: str or any
        """

        try:

            # Attempt to retrieve the value under the given section and name.
            x = self.parser.get(section, name)
            return x

        except (NoOptionError, NoSectionError):

            # If the value is missing, but a default has been provided:
            if fallback is not None:
                # Return the default with a warning (if requested).
                self._missing_value_warning(section, name, fallback)
                return fallback

            # Otherwise, return an error if no default was provided.
            self._missing_value_error(section, name)

    def get_str(self, section: str, name: str, fallback: str = None) -> str:
        """
        This function will return the string value for a parameter in the
        selected configuration file, identified by its section and name. If
        the value is missing, an optional default value can be used instead.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: str

        :return: Returns either the string found within the config file
            for the given section and name, or the default value if not
            found.
        :rtype: str
        """

        # Return the value (or default if missing) as a string.
        return str(self._get_value(section, name, fallback))

    def get_str_alpha(self, section: str, name: str, fallback: str = None) -> str:
        """
        This function will return the alphanumeric string value for a 
        parameter in the selected configuration file, identified by its section 
        and name. If the value is missing, an optional default value can be
        used instead.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: str

        :return: Returns either the alphanumeric parts of the string found 
            within the config file for the given section and name, or the
            default value if not found.
        :rtype: str
        """
        value = str(self._get_value(section, name, fallback))
        return re.sub(r'[^a-z0-9]', '', value.strip().casefold())

    def get_int(self, section: str, name: str, fallback: int = None) -> int:
        """
        This function will return the integer value for a parameter in the
        selected configuration file, identified by its section and name.
        If the value is missing, an optional default value can be used instead.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: int

        :return: Returns either the integer value found within the config file
            for the given section and name, or the default value if not found.
        :rtype: int
        """

        # Perform evaluation of the returned string to support user
        # expressions. For example '1 / value', 'value ** 2' or '1e-6'.
        parsed_value = self._get_value(section, name, str(fallback))

        try:

            # Attempt to evaluate the string and convert it to a float.
            return int(eval(parsed_value))

        except Exception as e:

            # In the event of an error, display useful error information.
            raise NavConfigError(
                section, name, e.__class__.__name__,
                f"A valid integer value could not be "
                f"extracted from '{parsed_value}'!")

    def get_float(self, section: str, name: str, fallback: float = None) -> float:
        """
        This function will return the float value for a parameter in the
        selected configuration file, identified by its section and name.
        This function returns a numpy 64-bit float instead of a normal float.
        If the value is missing, an optional default value can be used instead.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: float

        :return: Returns either the float value found within the config file
            for the given section and name, or the default value if not found.
        :rtype: float
        """

        try:

            if fallback is not None:
                if not self.is_value_set(section, name):
                    return fallback

            parsed_value = self._get_value(section, name)
            return float(eval(parsed_value))

        except Exception as e:

            # Obtain the original input values
            parsed_value = self._get_value(section, name, str(fallback))

            # In the event of an error, display useful error information.
            raise NavConfigError(
                section, name, e.__class__.__name__,
                f"A valid float value could not be "
                f"extracted from '{parsed_value}'!")

        # Perform evaluation of the returned string to support user
        # expressions. For example '1 / value', 'value ** 2' or '1e-6'.
        # parsed_value = self._get_value(section, name, str(fallback))
        #
        # try:
        #
        #     # Attempt to evaluate the string and convert it to a float.
        #     return float(eval(parsed_value))
        #
        # except Exception as e:
        #
        #     # In the event of an error, display useful error information.
        #     raise NavConfigError(section, name, e.__class__.__name__,
        #                                 f"A valid float value could not be "
        #                                 f"extracted from '{parsed_value}'!")

    def get_bool(self, section: str, name: str, fallback: bool = None) -> bool:
        """
        This function will return the boolean value for a parameter in the
        selected configuration file, identified by its section and name.
        Acceptable True values are 'yes', 'true', 'on' and '1'. Acceptable
        False values are 'no', 'false', 'off' and '0'. If the value is missing,
        an optional default value can be used instead.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: bool

        :return: Returns either the boolean found within the config file for
            the given section and name, or the default value if not found.
        :rtype: bool
        """

        try:

            # Attempt to evaluate the string and convert it to a boolean.
            if fallback is not None:
                return self.parser.getboolean(
                    section, name, fallback=fallback)
            else:
                return self.parser.getboolean(section, name)

        except Exception as e:

            # Obtain the original input values
            parsed_value = self._get_value(section, name, str(fallback))

            # In the event of an error, display useful error information.
            raise NavConfigError(section, name, e.__class__.__name__,
                                        f"A valid boolean value could not be "
                                        f"extracted from '{parsed_value}'!")

    def get_csv_numeric(self, section: str,
                        name: str,
                        fallback: npt.ArrayLike = None,
                        sep: str = ",") -> npt.ArrayLike | None:
        """
        This function will return an array of values that are separated by a
        common character (comma by default) from the selected configuration
        file, identified by its section and name. This function returns a
        numpy array. If the value is missing, an optional default value can
        be used instead.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value(s) to use if not found.
        :type fallback: list

        :param sep: (Optional) The item separator character (default=',').
        :type: str

        :return: A numpy array containing the written items.
        :rtype: list
        """

        # Read the line from the file as a string.
        line_text = self._get_value(section, name, "")

        # If not empty:
        if line_text != "":

            # Return the parsed values as an array.
            # lines: list[str] = line_text.split(",")
            # return [s.strip() for s in lines]

            return np.fromstring(line_text, sep=sep)
        else:

            # Otherwise, return the default value.
            return fallback

    def get_csv_str(self, section: str,
                    name: str,
                    fallback: list = None,
                    sep: str = ",") -> list | None:
        """
        This function will return an array of string values that are
        separated by a common character (comma by default) from the selected
        configuration file, identified by its section and name. This function
        returns a list of one or more character values. If the value is
        missing, an optional default value can be used instead.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value(s) to use if not found.
        :type fallback: list | None

        :param sep: (Optional) The item separator character (default=',').
        :type: str

        :return: A Python list containing the written items.
        :rtype: list | None
        """


        # Read the line from the file as a string.
        line_text = self._get_value(section, name, "")

        # If not empty:
        if line_text != "":
            # Return the parsed values as an array.
            lines: list[str] = line_text.split(sep)
            return [s.strip() for s in lines]

        # Otherwise, return the default value.
        return fallback


    # def get_list_of_values(self, section: str,
    #                        name: str,
    #                        fallback: list = None,
    #                        sep: str = ",") -> list:
    #     """
    #     This function will return an array of string values that are
    #     separated by a common character (comma by default) from the selected
    #     configuration file, identified by its section and name. This function
    #     returns a list of one or more character values. If the value is
    #     missing, an optional default value can be used instead.
    #
    #     :param section: The section of the configuration file.
    #     :type section: str
    #
    #     :param name: The name of the variable to read from the given section.
    #     :type name: str
    #
    #     :param fallback: (Optional) The default value(s) to use if not found.
    #     :type fallback: list
    #
    #     :param sep: (Optional) The item separator character (default=',').
    #     :type: str
    #
    #     :return: A Python list containing the written items.
    #     :rtype: list
    #     """
    #
    #     # Read the line from the file as a string.
    #     line_text = self._get_value(section, name, "")
    #
    #     # If the string is not empty:
    #     if line_text != "":
    #
    #         # If more than one value exists:
    #         if sep in line_text:
    #
    #             # Split the items into a list.
    #             return line_text.split(sep)
    #         else:
    #
    #             # Return the single item as a list.
    #             return [line_text]
    #     else:
    #
    #         # Otherwise, return the default value.
    #         return fallback

    def get_value_filepath(self, section: str,
                           name: str,
                           fallback: str | Path = None,
                           must_exist: bool = False) -> Path:
        """
        This function will return a local filepath read for a parameter in the
        selected configuration file, identified by its section and name. This
        function will handle read paths in a platform independent manner,
        automatically deciding and converting when needed. This allows paths
        read from the configuration file to be written in either Linux/Unix or
        Windows/NT format without affecting behaviour on different platforms.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: str

        :param must_exist: (Optional) Should the path exist on the filesystem?
            If enabled an error will raise if the given path is not found on
            the system (default=True).

        :return: A valid filepath for the current system to the given
            location/resources.
        :rtype: Path
        """

        # Obtain the value from the configuration file. If the value
        # is not found and a default value if given, use that instead.
        given_path = self.get_str(section, name, fallback)

        # Use the path as a Linux filepath (default by pathlib).
        return_path = Path(given_path)

        # If the path does not exist, attempt to convert it from a
        # Windows path to the path format of the current system.
        if not return_path.exists():
            return_path = Path(PureWindowsPath(given_path))

        # If the path still cannot be found:
        if must_exist and not return_path.exists():

            # Return a constructed error.
            raise NavConfigError(
                section, name, "FileNotFoundError",
                f"The file path '{return_path}' does not exist!")

        # Finally, return the valid path in the current system's format.
        return return_path

    def get_value_filepath_str(self, section: str, name: str, fallback: str = None):
        """
        This function will return a local filepath read for a parameter in the
        selected configuration file, identified by its section and name. This
        function will handle read paths in a platform independent manner,
        automatically deciding and converting when needed. This allows paths
        read from the configuration file to be written in either Linux/Unix or
        Windows/NT format without affecting behaviour on different platforms.

        Unlike 'get_value_filepath' this function does not enforce the
        existence of a requested file/database_dir.

        :param section: The section of the configuration file.
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param fallback: (Optional) The default value to use if not found.
        :type fallback: str

        :return: A filepath for the current system to the given location/
            resources. This resource does not have to exist.
        :rtype: Path
        """

        # Initially, attempt to extract the corresponding as a string.
        given_path = self.get_str(section, name, fallback)

        # Return this value converted into the current systems
        # format (i.e. correct forward/backward slashes).
        return Path(PureWindowsPath(given_path))

    def is_value_set(self, section: list or str, value: list or str) -> bool:
        """
        Returns true if there is an entry for (all of) the given sections and
        values. This function works for both single values and lists of
        values. For multiple values, both lists must be of the same length.

        :param section: A single or list of sections names.
        :type section: list or str

        :param value: A single or list of value names.
        :type section: list or str

        :return: True if an entry could be found in the provided configuration
            for all the given section-list pairs. False if one or more entries
            could not be found.
        :rtype: bool
        """

        # Convert the input arguments to lists if they aren't already.
        section = (list([section]), section)[isinstance(section, list)]
        value = (list([value]), value)[isinstance(value, list)]

        # Ensure the given list lengths are the same.
        assert len(section) == len(value), \
            "The given section and value lists must be of same length!"

        # For each section-value pair:
        for sec, val in zip(section, value):

            # Check if an entry could be found in the configuration.
            if not self.parser.has_option(sec, val):
                # If not, return False.
                return False

        # Return True if an entry has been given for each section-value pair.
        return True

    def overwrite_value(self, section: str, name: str, new_value):
        """
        This simple function can be used to overwrite or insert any value
        extracted from the given configuration file. This is only to be used
        for automatically generated values and is not an alternative to
        fallbacks.

        :param section: The section name of the variable to override
        :type section: str

        :param name: The name of the variable to read from the given section.
        :type name: str

        :param new_value: The new value to use for the given section and name.
        :type new_value: str, int, float
        """

        # Set/override the value within the config parser.
        self.parser.set(section, name, str(new_value))

    def copy_file(self, dest: Path):
        """
        Copies the configuration file to a destination.
        This method produces a copy of the configuration file at the
        requested destination.

        :param dest: The location to write the file to.
        :type dest: Path
        """

        try:
            copyfile(self.config_file, dest)

        except OSError:
            warning_msg = "Unable to copy configuration file!"
            warn(warning_msg, NavConfigWarning)

    def write_file(self, dest: Path):
        """
        Writes the configuration values to a file.
        This method writes the configuration values in their current state
        (including modifications) to a requested file. The produced file can
        then be re-used as a settings file to repeat experimentation.

        :param dest: The location to write the file to.
        :type dest: Path
        """

        try:
            with open(dest, 'w') as config_file:
                self.parser.write(config_file)

        except OSError:
            warning_msg = "Unable to write configuration file!"
            warn(warning_msg, NavConfigWarning)


class NavConfigWarning(UserWarning):
    """
    A custom warning type used to inform the user about invalid, unsuitable
    or missing values present in the configuration file. If these warnings
    generated can be safely ignored, this warning type can be suppressed
    to hide all configuration file warnings for the session. Suppressing
    this warning type will not affect the displaying of other warnings.
    """
    pass


class NavConfigError(Exception):
    """
    A custom exception type used for inform the user about erroneous values
    present in the provided configuration file that cannot be ignored. This
    includes syntax, value type and conflicting value errors. This exception
    type is primarily used for formatting the errors being displayed. This
    custom class also allows configuration based errors to be distinguished
    from serious errors such faults not detected until after delivery.
    """

    # A short error code/description for the error.
    error_code = None

    # The section for the variable causing the error.
    section = None

    # The name of the variable causing the error.
    variable = None

    # A meaningful description of the error.
    desc = None

    # The error message to be printed in the output console.
    message = None

    def __init__(self, section_name: str, variable_name: str,
                 error_type: str, error_desc: str):
        """
        The constructor used for raising a new Quantum Navigation
        Configuration Error. This takes information about the configuration
        variable whose value has caused an error that cannot be overcome.

        :param section_name: The name of the section for the variable in the
            configuration file that is causing an error.
        :type section_name: str

        :param variable_name: The name of the variable in the configuration
            file that is causing an error.
        :type variable_name: str

        :param error_type: A short description about the error, such as
            'Must Be Positive' or 'Conflicting Value'.
        :type error_type: str

        :param error_desc: A description about the cause of the error. This
            message should explain why the value is invalid and how it can
            be changed to become valid.
        :type error_desc: str
        """

        # Record the section of the variable causing a fault.
        self.section = section_name

        # Record the name of the variable causing a fault.
        self.variable = variable_name

        # Record the short code/description of the error.
        self.error_code = error_type

        # Record a description of the error.
        self.desc = error_desc

        # Construct the exception message to be displayed.
        self.message = f"{self.error_code}\n\n" \
                       f"Config Section:\t\t'{self.section}'\n" \
                       f"Variable Name:\t\t'{self.variable}'\n\n" \
                       f"Error Description:\t{self.desc}"

        # Call the original exception constructor
        # with the assembled error message.
        super().__init__(self.message)

    def __str__(self):
        """
        This function overrides the default behaviour when the raised
        exception is converted to a string. This is useful as it
        simplifies the displaying of the error.

        :return: A string describing the exception raised.
        :rtype: str
        """

        # Extract the name of the current class.
        class_name = self.__class__.__name__

        # Append the class name to the error message.
        return f"{class_name}:\t{self.message}"


if __name__ == '__main__':
    pass