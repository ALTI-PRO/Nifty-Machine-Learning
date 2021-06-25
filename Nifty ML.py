import pandas as pd
import pandas_datareader.data as pdat
import datetime
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

start_date = datetime.datetime(2018, 1, 1) #YYY-MM-DD
end_date = datetime.datetime(2020, 12, 31)

nifty = pdat.DataReader("^NSEI", 'yahoo', start_date, end_date)

                                    #Function for calculating RSI indicator 
def rsi(d, p):
    
    temp = []
    rsii = []
    pog = []    
    nog = []
    nav = []
    pav = []
    RS = []
    
    for x in range (1, len(d)):
        temp.append(float(d[x]-d[x-1]))

    for x in range(0, p):
         if temp[x] < 0 :
                nog.append(temp[x])
                pog.append(0)

         else:
                pog.append(temp[x])
                nog.append(0)
                
    nav.append(abs(sum(nog)/p)) 
    pav.append(sum(pog)/p)
    
    for x in range(p+1, len(d)):
        pogx = []    
        nogx = []
        for y in range (x-p, x):
            if temp[y] < 0 :
                nogx.append(temp[y])
                pogx.append(0)

            else:
                pogx.append(temp[y])
                nogx.append(0)
  
        
           
        nav.append(((nav[x-(p+1)]*(p-1))+abs(nogx[p-1]))/p)
        pav.append( ((pav[x-(p+1)]*(p-1))+pogx[(p-1)])/p  ) 

    for x in range(len(nav)):
        RS.append(pav[x]/nav[x])
        rsii.append(100-(100/(1+RS[x])))
       
      
    return(rsii)

#Creating OHLC lists
open_p = nifty['Open']
high_p = nifty['High']
low_p = nifty['Low']
close_p = nifty['Adj Close']


#Calculating all the indicators 

period = 14
rs = []
hlp = []
pc = []
nc_temp = ['nan']
cp = [x for x in close_p]

[rs.append('nan') for i in range (period)]
rs_temp = rsi(close_p, period)
rsi_values = rs + rs_temp #RSI calculation

moving_avg = close_p.rolling(window=period).mean() #moving average calculation

[hlp.append( ((high_p[i] - low_p[i])/close_p[i])*100 ) for i in range (len(close_p))] # high-low range percentage

[pc.append( ((close_p[i] - open_p[i])/open_p[i])*100 ) for i in range (len(close_p))]

next_close = nc_temp + cp


#Features and dependent variable dataframe

df = pd.DataFrame(list(zip(rsi_values, moving_avg, hlp, pc, close_p, next_close )),
               columns =['RSI', 'SMA', 'HLP', 'PC', 'Close', 'Next Day Close'])

X = df.iloc[period:, :-2].values #Independent Variables
y = df.iloc[period:, 5].values   #Dependent Variable

#Splitting the data into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#MULTIPLE LINEAR REGRESSION 

#Training the on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print (   np.concatenate(  (  y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)  ), 1  )  )


r2_score(y_test, y_pred) #Multiple Linear Regression Model Performance 

#Random Forest Regression
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

r2_score(y_test, y_pred) #Random Forest Regression Model Performance
