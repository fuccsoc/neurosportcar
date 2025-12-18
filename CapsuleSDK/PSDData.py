import ctypes
from CapsuleSDK.Error import Error, Error_Code, CapsuleException

class PSDData_Band(ctypes.c_int):
    PSDData_Band_Delta = 0
    PSDData_Band_Theta = 1
    PSDData_Band_Alpha = 2
    PSDData_Band_SMR = 3
    PSDData_Band_Beta = 4

class PSDData:
    def __init__(self, pointer, lib):
        self._lib = lib
        self._pointer = pointer

        self._lib.clCPSDData_GetTimestampMilli.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]
        self._lib.clCPSDData_GetTimestampMilli.restype = ctypes.c_uint64

        self._lib.clCPSDData_GetFrequenciesCount.restype = ctypes.c_int
        self._lib.clCPSDData_GetFrequenciesCount.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]

        self._lib.clCPSDData_GetChannelsCount.restype = ctypes.c_int
        self._lib.clCPSDData_GetChannelsCount.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]

        self._lib.clCPSDData_GetFrequency.restype = ctypes.c_double
        self._lib.clCPSDData_GetFrequency.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(Error)]

        self._lib.clCPSDData_GetPSD.restype = ctypes.c_double
        self._lib.clCPSDData_GetPSD.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(Error)]

        self._lib.clCPSDData_GetBandUpper.restype = ctypes.c_float
        self._lib.clCPSDData_GetBandUpper.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(Error)]

        self._lib.clCPSDData_GetBandLower.restype = ctypes.c_float
        self._lib.clCPSDData_GetBandLower.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(Error)]
        
        self._lib.clCPSDData_HasIndividualAlpha.restype = ctypes.c_bool
        self._lib.clCPSDData_HasIndividualAlpha.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]
        self._lib.clCPSDData_GetIndividualAlphaLower.restype = ctypes.c_float
        self._lib.clCPSDData_GetIndividualAlphaLower.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]
        self._lib.clCPSDData_GetIndividualAlphaUpper.restype = ctypes.c_float
        self._lib.clCPSDData_GetIndividualAlphaUpper.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]
        self._lib.clCPSDData_HasIndividualBeta.restype = ctypes.c_bool
        self._lib.clCPSDData_HasIndividualBeta.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]
        self._lib.clCPSDData_GetIndividualBetaLower.restype = ctypes.c_float
        self._lib.clCPSDData_GetIndividualBetaLower.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]
        self._lib.clCPSDData_GetIndividualBetaUpper.restype = ctypes.c_float
        self._lib.clCPSDData_GetIndividualBetaUpper.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(Error)]
    
    def get_timestamp(self):
        error = Error()
        ts = self._lib.clCPSDData_GetTimestampMilli(self._pointer, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return ts

    def get_frequencies_count(self):
        error = Error()
        count = self._lib.clCPSDData_GetFrequenciesCount(self._pointer, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        
        return count

    def get_channels_count(self):
        error = Error()
        count = self._lib.clCPSDData_GetChannelsCount(self._pointer, error)
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        
        return count

    def get_frequency(self, idx):
        error = Error()
        freq = self._lib.clCPSDData_GetFrequency(self._pointer, idx, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        
        return freq

    def get_psd(self, channelIdx, freqIdx):
        error = Error()
        psd = self._lib.clCPSDData_GetPSD(self._pointer, channelIdx, freqIdx, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)
        
        return psd

    def get_band_upper(self, band : PSDData_Band):
        error = Error()
        band = self._lib.clCPSDData_GetBandUpper(self._pointer, band, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return band

    def get_band_lower(self, band : PSDData_Band):
        error = Error()
        band = self._lib.clCPSDData_GetBandLower(self._pointer, band, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return band

    def has_individual_alpha(self):
        error = Error()
        ans = self._lib.clCPSDData_HasIndividualAlpha(self._pointer, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return ans

    def get_alpha_lower(self):
        error = Error()
        alpha = self._lib.clCPSDData_GetIndividualAlphaLower(self._pointer, error)
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return alpha
        
    def get_alpha_upper(self):
        error = Error()
        alpha = self._lib.clCPSDData_GetIndividualAlphaUpper(self._pointer, error)
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return alpha

    def has_individual_beta(self):
        error = Error()
        ans = self._lib.clCPSDData_HasIndividualBeta(self._pointer, ctypes.byref(error))
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return ans

    def get_beta_lower(self):
        error = Error()
        alpha = self._lib.clCPSDData_GetIndividualBetaLower(self._pointer, error)
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return alpha
        
    def get_beta_upper(self):
        error = Error()
        alpha = self._lib.clCPSDData_GetIndividualBetaUpper(self._pointer, error)
        if error.code is not Error_Code.OK:
            raise CapsuleException(error)

        return alpha
