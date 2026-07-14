
import json
import struct
import tempfile
import weakref
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
        self._exported_views = []
        self._mapping_is_destination = False
        self._backing_file = None
        self._retired_mappings = []

        if auto_load and file_path.exists():
            self.read()

    def _live_exported_views(self) -> bool:
        self._exported_views = [
            reference for reference in self._exported_views
            if reference() is not None
        ]
        return bool(self._exported_views)

    @staticmethod
    def _close_mapping(mapping, backing_file) -> None:
        mapped = getattr(mapping, "_mmap", None)
        if mapped is not None:
            mapped.close()
        if backing_file is not None:
            backing_file.close()

    def _cleanup_retired_mappings(self) -> None:
        retained = []
        for mapping, backing_file, references, is_destination in self._retired_mappings:
            references = [reference for reference in references
                          if reference() is not None]
            if references:
                retained.append(
                    (mapping, backing_file, references, is_destination))
            else:
                self._close_mapping(mapping, backing_file)
        self._retired_mappings = retained

    def _release_memmap(self, *, for_overwrite: bool = False) -> None:
        """Release an unshared mapping before replacing this table's data.

        Arrays returned by :meth:`get_data` may outlive the table's current
        view. Closing their shared mapping would leave those arrays pointing
        at invalid memory. Reads can simply leave such an old mapping alive;
        overwrites fail explicitly until the caller releases the views.
        """
        self._cleanup_retired_mappings()
        if for_overwrite and any(
                is_destination for _, _, _, is_destination
                in self._retired_mappings):
            raise RuntimeError(
                "Cannot overwrite results while get_data() views are retained")
        data = self._data_table
        if isinstance(data, np.memmap):
            if self._live_exported_views():
                if for_overwrite and self._mapping_is_destination:
                    raise RuntimeError(
                        "Cannot overwrite results while get_data() views are retained")
                self._retired_mappings.append((
                    data, self._backing_file, self._exported_views,
                    self._mapping_is_destination))
            else:
                self._close_mapping(data, self._backing_file)
            self._time_steps = None
        self._data_table = None
        self._backing_file = None
        self._mapping_is_destination = False
        self._exported_views = []

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
            self._mapping_is_destination = True

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

        self._release_memmap(for_overwrite=True)

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

        # Keep the writer's queryable view on an anonymous temporary mapping.
        # This avoids a dense RAM copy without leaving the destination file
        # locked against another ResultsTable writer on Windows.
        backing_file = tempfile.TemporaryFile()
        backing_file.truncate(int(np.prod(data_size)) * np.dtype(np.float64).itemsize)
        loaded = np.memmap(
            backing_file, np.float64, 'r+', shape=data_size)
        loaded[:] = to_write
        loaded.flush()
        self._close_mapping(to_write, None)

        self._description = desc
        self._time_steps = loaded[:, 0]
        self._data_table = loaded
        self._index_dict = fields
        self._backing_file = backing_file
        self._mapping_is_destination = False

    def get_data(self) -> dict:

        if self._data_table is None:
            raise LookupError("no data within table")

        values_view = self._data_table.view()
        time_view = self._time_steps.view()
        fields = {}
        columns = ["time"] * values_view.shape[1]

        for field, (index, size) in self._index_dict.items():

            if size == 1:
                fields[field] = values_view[:, index]
                columns[index] = field

            else:
                fields[field] = values_view[:, index:index + size]
                for i in range(index, index + size):
                    columns[i] = f"{field} ({i - index + 1})"

        table = {
            'columns': columns,
            'values': values_view
        }

        if isinstance(self._data_table, np.memmap):
            for view in (time_view, values_view, *fields.values()):
                self._exported_views.append(weakref.ref(view))

        return {
            'description': self._description,
            'time_steps': time_view,
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


