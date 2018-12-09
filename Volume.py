class VolumeSignals():

    def __init__(self):

        self.selected_signal = None


    def EOM(self, high, low, close, volume, n):

        self.selected_signal = 'EOM'

        eom_series = pd.DataFrame({"Price" : np.round(close, 2)})

        k = 100000000

        distance_moved = ((high + low) / 2) - ((high.shift() + low.shift()) / 2)

        box_ratio = (volume / k) / (high - low)

        eom_series['EOM'] = np.round((distance_moved / box_ratio), 3)

        eom_series['EOM_AVG'] = np.round(eom_series['EOM'].rolling(n).mean(), 3)

        return(eom_series)


    def PVT(self, close, volume, n):

        self.selected_signal = 'Price Volume Trend'

        pvt_series = pd.DataFrame({"Price" : close})

        pvt_series['PVT'] = ((close.diff()) / (close.shift(1) * volume))

        pvt_series['PVT_AVG'] = pvt_series['PVT'].rolling(n).sum()

        return(pvt_series)


    def VOLUME(self, close, high, low, volume, n):

        self.selected_signal = 'Volume'

        oscillator = volume * (close - (high + low) / 2)

        cummulative_sum = oscillator.cumsum()

        rolling_sum = oscillator.rolling(n).sum()

        return(cummulative_sum, rolling_sum)


    def VO(self, volume, fast_n, slow_n):

        self.selected_signal = 'Volume Oscillator'

        fast_average = volume.rolling(fast_n).mean()

        slow_average = volume.rolling(slow_n).mean()

        diff = fast_average - slow_average

        oscillator = (diff / slow_average) * 100

        return(oscillator)
