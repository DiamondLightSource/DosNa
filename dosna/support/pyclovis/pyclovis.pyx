import json
import logging

from cpython cimport PyObject

cdef extern from "Python.h":
    PyObject *PyBytes_FromStringAndSize(char *v, Py_ssize_t len) except NULL
    char* PyBytes_AsString(PyObject *string) except NULL

cdef extern from "clovis_functions.h":
    ctypedef unsigned long uint64_t
    int init_clovis(char * laddr, char * ha_addr, char * prof_id,
                    char * proc_fid, size_t block_size, unsigned int tier)
    int write_object(uint64_t high_id, uint64_t low_id,
                     char *buffer, size_t length)
    int read_object(uint64_t high_id, uint64_t low_id, char *buffer,
                    size_t length)
    int exist_object(uint64_t high_id, uint64_t low_id)
    int create_object(uint64_t high_id, uint64_t low_id)
    int delete_object(uint64_t high_id, uint64_t low_id)
    void fini_clovis()

cdef:
    uint64_t _METADATA_CHUNK_ID = 0xffffffffffffffff

REQUIRED_OPTIONS = ['laddr', 'ha_addr', 'prof_id', 'proc_fid', 'block_size', 'tier']

log = logging.getLogger(__name__)


class ClovisConnectionNotInitialised(Exception):
    pass


class ClovisOptionRequired(Exception):
    pass


class Clovis:
    def __init__(self, conffile=None):
        self._process_parameters(conffile)
        self.is_connected = False

    def _process_parameters(self, conffile="clovis.conf"):
        self.options = {}
        self.conffile = conffile
        if conffile is not None:
            self._process_config_file(conffile)

    def _check_connected(self):
        if not self.connected:
            raise ClovisConnectionNotInitialised()

    def _process_config_file(self, conffile):
        # parse key=val lines
        with file(conffile) as file_handle:
            for line in file_handle:
                if "=" not in line or line.startswith("#"):
                    continue
                key, val = line.split('=')
                self.options[key.strip()] = val.strip()
        self._validate_options()
        self.options['block_size'] = int(self.options['block_size'])
        self.options['tier'] = int(self.options['tier'])

    def _validate_options(self):
        for option in REQUIRED_OPTIONS:
            if option not in self.options:
                raise ClovisOptionRequired('Clovis: option {} needed'
                                           .format(option))

    def get_option(self, key):
        return self.options.get(key)

    def _get_app_id(self, name):
        app_id = hash(name) & 0xffffffff

        if app_id == 0:
            # make sure id will be bigger than M0_CLOVIS_ID_APP
            app_id = 1

        return app_id

    def connect(self):
        log.info("Initializing clovis with options: %s", self.options)
        rc = init_clovis(self.options['laddr'], self.options['ha_addr'],
                         self.options['prof_id'], self.options['proc_fid'],
                         self.options['block_size'], self.options['tier'])

        if rc:
            raise Exception("Error {} while initialising Clovis".format(rc))

        self.is_connected = True

    @property
    def connected(self):
        return self.is_connected

    def read_object_chunk(self, name, uint64_t chunk_id, size_t length):
        cdef:
            uint64_t app_id

        app_id = self._get_app_id(name)
        data = self.read_object_by_id(app_id, chunk_id, length)
        return data

    def write_object_chunk(self, name, uint64_t chunk_id, data, size_t length):
        cdef:
            uint64_t app_id

        app_id = self._get_app_id(name)
        self.write_object_by_id(app_id, chunk_id, data, length)

    def has_object_chunk(self, name, uint64_t chunk_id):
        cdef:
            uint64_t app_id

        app_id = self._get_app_id(name)
        return self.has_object_by_id(app_id, chunk_id)

    def create_object_chunk(self, name, uint64_t chunk_id):
        cdef:
            uint64_t app_id

        app_id = self._get_app_id(name)
        self.create_object_by_id(app_id, chunk_id)

    def delete_object_chunk(self, name, uint64_t chunk_id):
        cdef:
            uint64_t app_id

        app_id = self._get_app_id(name)
        self.delete_object_by_id(app_id, chunk_id)

    # Current implementation of metadata
    # uses object with low_id = _METADATA_CHUNK_ID
    # this may change in future as clovis provides metadata support
    def has_object_metadata(self, name):
        return self.has_object_chunk(name, _METADATA_CHUNK_ID)

    def get_object_metadata(self, name):
        metadata = self.read_object_chunk(name, _METADATA_CHUNK_ID,
                                          self.options['block_size'])
        return json.loads(metadata.rstrip("\0x0"))

    def set_object_metadata(self, name, value):
        data = json.dumps(value)
        if len(data) > self.options['block_size']:
            raise Exception("Metadata shouldn't be bigger than block size: "
                            "{} > {}".format(len(value),
                                             self.options['block_size']))
        self.write_object_chunk(name, _METADATA_CHUNK_ID, data, len(data))

    def create_object_metadata(self, name):
        cdef:
            uint64_t app_id

        app_id = self._get_app_id(name)
        self.create_object_by_id(app_id,  _METADATA_CHUNK_ID)

    def delete_object_metadata(self, name):
        cdef:
            uint64_t app_id

        app_id = self._get_app_id(name)
        self.delete_object_by_id(app_id,  _METADATA_CHUNK_ID)

    def has_object_by_id(self, uint64_t high_id, uint64_t low_id):
        self._check_connected()
        return bool(exist_object(high_id, low_id))

    def create_object_by_id(self, uint64_t high_id, uint64_t low_id):
        cdef:
            int rc

        self._check_connected()
        rc = create_object(high_id, low_id)
        if rc != 0:
            raise Exception("Error {} while creating object".format(rc))

    def delete_object_by_id(self, uint64_t high_id, uint64_t low_id):
        cdef:
            int rc

        self._check_connected()
        rc = delete_object(high_id, low_id)
        if rc < 0:  # object doesn't exist
            raise KeyError("Object not found")

        if rc != 0:
            raise Exception("Error {} while deleting object".format(rc))

    def write_object_by_id(self, uint64_t high_id, uint64_t low_id,
                           char *buffer, size_t length):
        cdef:
            int rc

        self._check_connected()
        rc = write_object(high_id, low_id, buffer, length)
        if rc != 0:
            raise Exception("Error {} while writing object".format(rc))

    def read_object_by_id(self, uint64_t high_id, uint64_t low_id,
                          size_t length):
        cdef:
            char *buffer = NULL
            PyObject *buffer_bytes
            int rc

        self._check_connected()
        buffer_bytes = PyBytes_FromStringAndSize(NULL, length)
        buffer = PyBytes_AsString(buffer_bytes)
        rc = read_object(high_id, low_id, buffer, length)
        if rc != 0:
            raise Exception("Error {} while reading object".format(rc))
        return <object> buffer_bytes

    def disconnect(self):
        self.is_connected = False
        fini_clovis()
