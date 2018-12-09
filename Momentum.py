class MomentumOscillators():

    def __init__(self):

        self.selected_signal = None



    def CMO(self, close, n):

        price = np.array(close)

        pos = [];  neg = []

        for i in range(1, len(price)):

            if (price[i] - price[i-1]) > 0:

                pos.append(price[i] - price[i-1])

                neg.append(0)

            if (price[i] - price[i-1]) < 0:

                neg.append(abs(price[i] - price[i-1]))

                pos.append(0)

        pos_sum = [np.mean(pos[i:i+n]) for i in range(0, len(pos) -n+1)]

        neg_sum = [np.mean(neg[i:i+n]) for i in range(0, len(neg) -n+1)]

        momentum = np.divide(np.subtract(pos_sum, neg_sum), np.add(pos_sum, neg_sum)) * 100

        return(momentum)




    def PMO(self, close, n):

        decay = (2 / (n + 1))

        df = pd.DataFrame({"Price" : np.round(close, 2)})

        df['ROC'] = np.round((((close - close.shift(n)) / close.shift(n)) * 100), 3)

        df['PMO'] = np.round(df['ROC'].ewm(span= n).mean(), 3)

        df['LONG'] = ((df['PMO'] > df['PMO'].shift(1)) & (df['PMO'].shift(1) > df['PMO'].shift(2)))

        return(pmo)


    def ROC(close, n):

        df = pd.DataFrame({"Close" : close})

        df['ROC'] = np.multiply(np.divide(np.subtract(df['Close'], df['Close'].shift(n)), df['Close'].shift(n)), 100)

        df['ROC-Delta'] = np.log(df['ROC'] / df['ROC'].shift(5)) * np.exp(n**5)


        return(df)



    def RSI(close, n, n_smooth, n_roc):

        rsi_series = pd.DataFrame({"Price" : np.round(close, 2)})

        try:

            diff = np.diff(close)

            seed = diff[:n+1]

            up = seed[seed>=0].sum() /n

            down = -seed[seed<0].sum() /n

            rs = (up / down)

            rsi = np.zeros_like(close)

            rsi[:n] = 100 - (100 / (1 + rs))

        except ZeroDivisionError:

            np.seterr(divide= 'ignore')

        for i in range(n, len(close)):

            delta = diff[i-1]

            if delta > 0:

                upval = delta

                downval = 0

            else:

                upval = 0

                downval = -delta

            try:

                up = (up * (n - 1) + upval) /n

                down = (down * (n - 1) + downval) /n

                rs = up/down

                rsi[i] = 100 - (100 / (1 + rs))

            except ZeroDivisionError:

                np.seterr(divide= 'ignore')

        rsi_series['RSI'] = np.round(rsi, 2)

        ift_filter = (rsi_series['RSI'] - 50) * 0.1

        rsi_series['IFT'] = np.round((np.exp(ift_filter * 2) - 1) / (np.exp(ift_filter * 2) +1), 3)

        rsi_series['SMOOTH'] = np.round(rsi_series['RSI'].rolling(n_smooth).mean(), 2)

        rsi_series['ROC'] = (((rsi_series['RSI'] - rsi_series['RSI'].shift(n_roc)) / rsi_series['RSI']) * 10)

        return(rsi_series)




    def STOCH(high, low, close, n, k_smooth, d_smooth):

        '''
        --------------------
        Required Bar Data (OHLC)

        n = number of periods of previous datapoints to filter through oscillator

        k_smooth = First smoothing of the stochastic fast_k

        d_smooth = Second smoothing of the stochastic slow_k

        '''

        df = pd.DataFrame({"High" : np.round(high, 2), "Low" : np.round(low, 2), "Close" : np.round(close, 2)})

        min = low.rolling(n).min()

        max = high.rolling(n).max()

        df['Fast_K'] = np.round(100 * (close - min) / (max - min), 3)

        df['Slow_K'] = np.round(df['Fast_K'].rolling(k_smooth).mean(), 3)

        df['Slow_D'] = np.round(df['Slow_K'].rolling(d_smooth).mean(), 3)

        df['LONG'] = ((df['Slow_K'] < 10) & (df['Fast_K'] > df['Fast_K'].shift(1)))

        df['SHORT'] = ((df['Slow_K'] > 90) & (df['Fast_K'] < df['Fast_K'].shift(1)))

        return(df)



    def TRIX(close, k_1, k_2, k_3):

        self.selected_signal = 'Triple Smoothed EMA Oscillator'

        df = pd.DataFrame({"Price" : close})

        df['EMA-1'] = df['Price'].ewm(span= k_1).mean()

        df['EMA-2'] = df['EMA-1'].ewm(span= k_2).mean()

        df['EMA-3'] = df['EMA-2'].ewm(span= k_3).mean()

        df['TRIX'] = (((df['EMA-3'] - df['EMA-3'].shift(1)) / df['EMA-3']) * 10000)

        return(df)



    def TSI(close, k_1, k_2):

        '''
        ----------------------------------------------------------
        True Strength Indicator
            
        Variables Needed: 
            
        close = list(closing prices)       
        
        k_1 = (2 / (1 + n)          
        
        k_2 = (2 / (1 + n) 

        n = Number of Periods for which to exponentially solve for                                                   
        ------------------------------------------------------------
        '''     

        self.selected_signal = 'True Strength Indicator'

        df = pd.DataFrame({"Price" : close})

        diff = close.diff()

        abs_diff = abs(close.diff())

        ema_1 = diff.ewm(span= k_1).mean() 

        abs_ema_1 = abs_diff.ewm(span= k_1).mean()

        ema_2 = ema_1.ewm(span= k_2).mean()

        abs_ema_2 = abs_ema_1.ewm(span= k_2).mean()

        df['TSI'] = (ema_2 / abs_ema_2) * 100

        return(df)




    def UO(close, short_n, med_n, long_n, smooth_n):

        self.selected_signal = 'Ultimate Oscillator'

        self.selected_signal = 'Ultimate Oscillator'

        buy_pressure = close - np.minimum(low, close.shift())

        buy_pressure = pd.Series(buy_pressure)

        true_range = np.maximum(high, close.shift()) - (np.minimum(low, close.shift()))

        true_range = pd.Series(true_range)

        short_average = buy_pressure.rolling(short_n).sum() / true_range.rolling(short_n).sum()

        med_average = buy_pressure.rolling(med_n).sum() / true_range.rolling(med_n).sum()

        long_average = buy_pressure.rolling(long_n).sum() / true_range.rolling(long_n).sum()

        oscillator = ((short_average * 4) + (med_average * 2) + (long_average * 2)) / 7

        smoothed_oscillator = oscillator.rolling(smooth_n).mean()

        return(oscillator, smoothed_oscillator)
