import tempfile
import mmap
import os
import sys
try:
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
except ImportError:
    np = None


valid_filemodes = ["r", "c", "r+", "w+"]

mode_equivalents = {
    "readonly": "r",
    "copyonwrite": "c",
    "readwrite": "r+",
    "write": "w+"
}


class MmapInfo(object):
    """Memmory mapping information for a data buffer of a shareable array"""
    # This class could be a namedtuple once Python 2.5 compat is dropped.

    def __init__(self, buffer, filename, offset, mode):
        self.buffer = buffer
        self.filename = filename
        self.offset = offset
        self.mode = mode

    def __repr__(self):
        return "MmapInfo(%r, %r, %r, %r)" % (
            self.buffer, self.filename, self.offset, self.mode)


def get_mmap_info(a):
    """Recursively look up the backing mmap info base if any.

    Return an MmapInfo(buffer, filename, offset, mode) instance or None.
    """
    b = getattr(a, 'base', None)
    if b is None:
        # a is not an array view derived from a mmap buffer
        return None
    elif isinstance(b, mmap.mmap):
        if np is not None and isinstance(a, np.memmap):
            # a is already a real memmap instance carrying metadata itself.
            return MmapInfo(b, a.filename, a.offset, a.mode)
        elif (hasattr(b, 'filename')
              and hasattr(b, 'offset')
              and hasattr(b, 'mode')):
            # Joblib provided a subclass of mmap that preserves metadata
            return MmapInfo(b, b.filename, b.offset, b.mode)
        else:
            # In numpy 1.7, the base collapsing mechanism makes
            # some array views on memmap instances loose the original
            # memmap filename and offset info. In this case treat them as
            # if this was an in-memory array and loose the shared memory
            # optimization
            return None
    else:
        # Recursive exploration of the base ancestry for numpy < 1.7
        return get_mmap_info(b)


def _find_offset(array, mmap_info=None):
    """Compute the total buffer offset for an mmap backed array"""
    # offset that comes from the striding differences between array and buffer
    if mmap_info is None:
        mmap_info = get_mmap_info(array)
    array_start = np.byte_bounds(array)[0]
    buffer_array = np.frombuffer(
        mmap_info.buffer, dtype=array.dtype, offset=mmap_info.offset)
    buffer_start = np.byte_bounds(buffer_array)[0]
    buffer_offset = array_start - buffer_start

    # offset from the backing memmap
    return buffer_offset + mmap_info.offset


def has_shareable_memory(a):
    """Return True if a is backed by a mmap buffer with map metadata.

    This functions check whether a's data is allocated in a memory mapped
    buffer and that the filename and other information are available for being
    able to reconstruct it as shared memory arrays in joblib.Parallel worker
    processes.

    Note: sliced views on numpy.memmap instances are not shareable
    despite being backed by a mmap buffer as the filename and offset
    information are lost during slicing. To prevent this, use
    joblib.numpy_mmap.mmap_array instead of numpy.memmap.

    """
    return get_mmap_info(a) is not None


class FileBackedMmapBuffer(mmap.mmap):
    """mmap subclass that stores filename and offset metadata for pickling"""

    def __new__(subtype, file_, length, mode='r+',
                offset=0, collect_file_on_gc=False):
        # we need a metaclass constructor as mmap.mmap is a builtin class
        # hence we would not be able to override the constructor parameters
        # with simple derivation
        if mode == 'c':
            access = mmap.ACCESS_COPY
        elif mode == 'r':
            access = mmap.ACCESS_READ
        else:
            access = mmap.ACCESS_WRITE

        # Offset for mmap can only handle some corse grain granularity (typicaly
        # 4096 bytes) hence the absolute offset is split into an mmap handled
        # part and the remaining will be have to be handled by the wrapping
        # datastructure such as a numpy array if any
        offset_mmap = offset - offset % mmap.ALLOCATIONGRANULARITY
        length -= offset_mmap
        remaining_offset = offset - offset_mmap

        self = mmap.mmap.__new__(subtype,
            file_.fileno(), length, access=access, offset=offset_mmap)
        self.offset = offset
        self.filename = file_.name
        self.mode = mode
        self.collect_file_on_gc = collect_file_on_gc
        self.remaining_offset = remaining_offset
        return self

    def __del__(self):
        if self.collect_file_on_gc and os.path.exists(self.filename):
            self.close()
            os.unlink(self.filename)


def mmap_array(filename=None, dtype=None, mode='r+', offset=0,
               shape=None, strides=None, order='C', temp_folder=None):
    """Create a mmap backed numpy.ndarray.

    This is an alternative to numpy.memmap for handling shared memory while
    making it robust to identify whether or not derived arrays such as sliced
    views are backed by mmap buffer that carries it's own filename and offset
    metadata to make it possible to override pickling efficiently (without
    memory copy) in a multiprocessing context.

    """
    if np is None:
        return None
    if dtype is None:
        dtype = np.uint8
    try:
        mode = mode_equivalents[mode]
    except KeyError:
        if mode not in valid_filemodes:
            raise ValueError("mode must be one of %s" %
                             (valid_filemodes + mode_equivalents.keys()))

    if mode == 'w+' and shape is None:
        raise ValueError("shape must be given when mode is 'w+'")

    if filename is None and strides is not None:
        raise ValueError('strides should be None if filename is None')

    # Handle both filenames, pre-opened file objects and temporary files to be
    # garbage collected when the buffer is gc'd
    file_mode = (mode == 'c' and 'r' or mode) + 'b'
    if filename is None:
        fd, filename = tempfile.mkstemp(
            prefix='joblib.mmap.temp_', dir=temp_folder)
        # We need to open a file object to zero the file to the right
        # size hence we can close the original fd
        file_ = open(filename, file_mode)
        os.close(fd)
        is_temporary, own_file = True, True
    elif hasattr(filename, 'read'):
        file_ = filename
        is_temporary, own_file = False, False
    else:
        file_ = open(filename, file_mode)
        is_temporary, own_file = False, True

    file_.seek(0, 2)
    file_length = file_.tell()
    dtype = np.dtype(dtype)

    if shape is None:
        n_bytes = file_length - offset
        if (n_bytes % dtype.itermsize):
            if own_file:
                file_.close()
            if is_temp:
                os.unlink(filename)
            raise ValueError("Size of available data is not a "
                             "multiple of the data-type size.")
        size = n_bytes // dtype.itemsize
        shape = (size,)
    else:
        if not isinstance(shape, tuple):
            shape = (shape,)
        size = 1
        for k in shape:
            size *= k

    n_bytes = long(offset + size * dtype.itemsize)

    if mode == 'w+' or (mode == 'r+' and file_length < n_bytes):
        file_.seek(n_bytes - 1, 0)
        file_.write(np.compat.asbytes('\0'))
        file_.flush()


    if sys.version_info[:2] >= (2, 6):
        # The offset keyword in mmap.mmap needs Python >= 2.6
        buffer = FileBackedMmapBuffer(
            file_, n_bytes, mode=mode, offset=offset,
            collect_file_on_gc=is_temporary)
    elif offset != 0:
        raise RuntimeError("Offset is not supported in Python < 2.6.")
    else:
        buffer = FileBackedMmapBuffer(
            file_, n_bytes, mode=mode, collect_file_on_gc=is_temporary)

    if own_file:
        file_.close()

    array = np.ndarray(shape, dtype=dtype, offset=buffer.remaining_offset,
                       buffer=buffer, order=order)
    if strides is not None:
        array = as_strided(array, strides=strides)
    return array


def as_mmap_array(a, mmap_mode=None, temp_folder=None):
    """Map the array content onto a file for sharing memory with subprocesses

    A temporary file is created and automatically deleted when the
    array is garbage collected.

    Parameters
    ----------
    a: array-like
        If a is an in-memory array, it is dumped on a contiguous
        filesystem backed buffer using joblib.numpy_mmap.FileBackedMmapBuffer
        to make it possible to share the data with subprocesses.
        shape, dtype and fortran ordering (if any) are preserved.

        If a is already an array with a FileBackedMmapBuffer it is
        returned as this. If it is a numpy.memmap instance it is
        remmaped using FileBackedMmapBuffer or if mmap_mode is provided
        and does not match the original mode, a new FileBackedMmapBuffer
        is opened on the backing file.
    mmap_mode: 'r', 'c', 'r+', 'w+' or None
        Specify a specific memory mapping mode. See mmap_array docstring for
        details.
    temp_folder: string or None
        Path to an existing folder used for creating the tempory file. If None,
        the tempory folder from the Python tempfile module is used.
    """
    a = np.asarray(a)
    mmap_info = get_mmap_info(a)
    if mmap_info is None:
        # Dump to temporary file and mmap it using contiguous memory only
        # preserving fortran ordering if needed
        order = 'F' if a.flags['F_CONTIGUOUS'] else 'C'
        b = mmap_array(temp_folder=temp_folder, dtype=a.dtype, shape=a.shape,
                       order=order)
        b[:] = a
        return b
    elif (isinstance(mmap_info.buffer, FileBackedMmapBuffer)
          and (mmap_mode is None or mmap_mode == mmap_info.mode)):
        # a is already an array created by mmap_array with the right mmap mode
        return a
    else:
        # this is either a numpy.memmap instance or something similar, let's
        # remap it to use the same backing file
        mode = mmap_info.mode if mmap_mode is None else mmap_mode
        if mode == 'w+':
            # prevent zeroing the original data
            mode = 'r+'
        offset = _find_offset(a, mmap_info)
        return mmap_array(mmap_info.filename, dtype=a.dtype, shape=a.shape,
                          mode=mode, offset=offset, strides=a.strides)
