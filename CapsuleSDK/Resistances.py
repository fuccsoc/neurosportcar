import ctypes

class Resistances:
    def __init__(self, pointer, lib):
        self._lib = lib
        self._pointer = pointer
        self._lib.clCResistance_GetCount.restype = ctypes.c_int
        self._len = self._lib.clCResistance_GetCount(self._pointer)

    def __len__(self):
        return self._len

    def get_channel_name(self, idx):
        self._lib.clCResistance_GetChannelName.restype = ctypes.c_char_p
        return self._lib.clCResistance_GetChannelName(self._pointer, idx).decode('utf-8');
    
    def get_value(self, idx) -> float:
        self._lib.clCResistance_GetValue.restype = ctypes.c_float
        return self._lib.clCResistance_GetValue(self._pointer, idx)