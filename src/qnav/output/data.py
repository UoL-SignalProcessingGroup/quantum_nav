
import json
import struct
import numpy as np

from pathlib import Path


class ResultsTable:
    """
    Dynamic on-disk data container for results records.
    Provides a container that uses virtual-memory, storing the results on
    disk but still allowing fast querying and retrieval. This can be useful
    for a number of tasks, including conversion and report generation,
    without loading all data into memory.
    """

    # The on-disk location for the result data table.
    _file_path: Path

    # The mode to use for opening and accessing the file.
    _read_only: bool

    # The results description string contained in the metadata.
    _description: str = None

    # The time steps series, corresponding to each result
    _time_steps: np.ndarray = None

    # The data table containing all the results in virtual array format.
    _data_table: np.ndarray = None

    # A dictionary used for indexing columns to data
    _index_dict: dict[str, tuple[int, int]] = None


    def __init__(self, file_path: Path, read_only: bool = False, auto_load: bool = True):
        self._file_path = file_path
        self._read_only = read_only

        if auto_load and file_path.exists():
            self.read()

    def _release_memmap(self) -> None:
        """Release this table's Windows file mapping before replacement."""
        data = self._data_table
        if isinstance(data, np.memmap):
            self._time_steps = None
            mapping = getattr(data, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._data_table = None

    def read(self, virtual_memory: bool = True):

        self._release_memmap()

        # Open the file and attempt to read header:
        with self._file_path.open('rb') as f:
            offset_bytes = f.read(4)
            offset = int(struct.unpack('>I', offset_bytes)[0])
            header_bytes = f.read(offset - 4)

        # Attempt to unpack the header:
        header = json.loads(header_bytes.decode("utf-8"))
        self._description = header["description"]
        self._index_dict = header["fields"]
        shape = header["size"]

        if virtual_memory:
            self._data_table = np.memmap(
                self._file_path, np.float64, 'r',
                offset=offset, shape=shape)

        else:
            self._data_table = np.fromfile(
                self._file_path, np.float64, offset=offset
            ).reshape(shape)

        self._time_steps = self._data_table[:, 0]


    @staticmethod
    def __generate_header(desc: str, time_steps: np.ndarray, **data: np.ndarray) -> dict:

        if np.ndim(time_steps) != 1:
            raise ValueError("time_steps must be 1 dimensional")

        total_num_rows = len(time_steps)
        total_num_cols = 1
        fields = {}

        for k, v in data.items():

            num_cols = 1 if np.ndim(v) == 1 else np.shape(v)[1]
            num_rows = np.shape(v)[0]

            fields[k] = (total_num_cols, num_cols)
            total_num_cols += num_cols

            if np.ndim(v) > 2:
                raise ValueError(f'data field "{k}" has more than 2 dimensions')

            if num_rows != total_num_rows:
                raise ValueError(f'data field "{k}" has {num_rows} rows, not {total_num_rows}')

        data_size = (total_num_rows, total_num_cols)

        return {
            "description": desc,
            "fields": fields,
            "size": data_size,
            "format_version": 1
        }


    def write(self, desc: str, time_steps: np.ndarray, **data: np.ndarray):

        if self._read_only:
            raise ValueError("Cannot write data in read-only mode")

        self._release_memmap()

        header = self.__generate_header(desc, time_steps, **data)
        data_size = header["size"]
        fields = header["fields"]

        # Encode the header contents in binary
        header_bytes = json.dumps(header).encode("utf-8")
        offset = len(header_bytes) + 4
        offset_bytes = struct.pack('>I', offset)  # Big-endian uint-1

        # Write the header data
        with self._file_path.open('wb') as f:
            f.write(offset_bytes)
            f.write(header_bytes)

        to_write = np.memmap(self._file_path, np.float64, 'r+',
                             offset=offset, shape=data_size)

        to_write[:, 0] = time_steps
        for field, (index, size) in fields.items():

            if size == 1:
                to_write[:, index] = data[field]
            else:
                to_write[:, index:index + size] = data[field]

        to_write.flush()
        # Writers retain an in-memory view so they do not keep the destination
        # mapped and block a later atomic rerun/overwrite on Windows. Explicit
        # readers still use virtual memory by default.
        loaded = np.array(to_write)
        mapping = getattr(to_write, "_mmap", None)
        if mapping is not None:
            mapping.close()

        self._description = desc
        self._time_steps = loaded[:, 0]
        self._data_table = loaded
        self._index_dict = fields

    def get_data(self) -> dict:

        if self._data_table is None:
            raise LookupError("no data within table")

        fields = {}
        columns = ["time"] * self._data_table.shape[1]

        for field, (index, size) in self._index_dict.items():

            if size == 1:
                fields[field] = self._data_table[:, index]
                columns[index] = field

            else:
                fields[field] = self._data_table[:, index:index + size]
                for i in range(index, index + size):
                    columns[i] = f"{field} ({i - index + 1})"

        table = {
            'columns': columns,
            'values': self._data_table
        }

        return {
            'description': self._description,
            'time_steps': self._time_steps,
            'table': table,
            'data': fields,
        }

    @property
    def is_data_loaded(self) -> bool:
        """
        States data has already been loaded.

        :return: Returns True if data is loaded.
        :rtype: bool
        """
        return self._data_table is not None


