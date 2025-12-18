import ctypes

class PPGTimedData:
    def __init__(self, pointer, lib):
        self._lib = lib
        self._pointer = pointer
        self._lib.clCPPGTimedData_GetTimestampMilli.restype = ctypes.c_uint64
        self._lib.clCPPGTimedData_GetValue.restype = ctypes.c_float
        self._lib.clCPPGTimedData_GetCount.restype = ctypes.c_int
        self._len = self._lib.clCPPGTimedData_GetCount(self._pointer)

    def __len__(self):
        return self._len

    def get_value(self, idx):
        return self._lib.clCPPGTimedData_GetValue(self._pointer, idx)

    def get_timestamp(self, idx):
        return self._lib.clCPPGTimedData_GetTimestampMilli(self._pointer, idx)