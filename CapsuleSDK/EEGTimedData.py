import ctypes
from CapsuleSDK.Error import Error, CapsuleException, Error_Code

class EEGTimedData:
    def __init__(self, pointer, lib):
        self._lib = lib
        self._pointer = pointer
        self._lib.clCEEGTimedData_GetRawValue.restype = ctypes.c_float
        self._lib.clCEEGTimedData_GetProcessedValue.restype = ctypes.c_float
        self._lib.clCEEGTimedData_GetTimestampMilli.restype = ctypes.c_uint64

    def get_samples_count(self):
        error = Error()
        count = self._lib.clCEEGTimedData_GetSamplesCount(self._pointer, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        return count

    def get_channels_count(self):
        error = Error()
        count = self._lib.clCEEGTimedData_GetChannelsCount(self._pointer, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        return count

    def get_raw_value(self, channelIdx, sampleIdx):
        error = Error()
        value = self._lib.clCEEGTimedData_GetRawValue(self._pointer, channelIdx, sampleIdx, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        return value

    def get_processed_value(self, channelIdx, sampleIdx):
        error = Error()
        value = self._lib.clCEEGTimedData_GetProcessedValue(self._pointer, channelIdx, sampleIdx, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        return value

    def get_timestamp(self, sampleIdx):
        error = Error()
        ts = self._lib.clCEEGTimedData_GetTimestampMilli(self._pointer, sampleIdx, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        return ts