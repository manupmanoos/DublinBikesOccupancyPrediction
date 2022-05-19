import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import sklearn.linear_model as linear_model
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
import time
from sklearn.svm import SVR
from sklearn.preprocessing import normalize


df = pd.read_csv(r'dublinbikes_20200101_20200401.csv',parse_dates=[1])

stations_selected = ["PRINCES STREET / O'CONNELL STREET",'CHARLEVILLE ROAD']
df1 = df[df['NAME'].isin(stations_selected)]
indices_original= df1.index
df1 = df1.iloc[:,[0,1,5]]

print('Plot of Station 33')
stations_selected = ["PRINCES STREET / O'CONNELL STREET"]
df_test = df[df['NAME'].isin(stations_selected)]
df_test = df_test.iloc[:,[1,5]]
df_test.index = df_test['TIME']
df_test = df_test.iloc[:,1]
plt.figure(figsize=(12,4))
df_test.plot()
plt.xlabel('Date')
plt.ylabel('Occupancy')
plt.show()

print('Plot of Station 107')
stations_selected = ['CHARLEVILLE ROAD']
df_test = df[df['NAME'].isin(stations_selected)]
df_test = df_test.iloc[:,[1,5]]
df_test.index = df_test['TIME']
df_test = df_test.iloc[:,1]
plt.figure(figsize=(12,4))
df_test.plot()
plt.xlabel('Date')
plt.ylabel('Occupancy')
plt.show()

unique_values = pd.DataFrame(df1['STATION ID'].unique())
encoded_values = pd.get_dummies(unique_values[0], prefix="STATION")
encoded_arr = pd.merge(left=unique_values,right=encoded_values,left_index=True,right_index = True)
encoded_arr = encoded_arr.rename(columns={0:"STATION ID"})
df2 = pd.merge(left=df1,right=encoded_arr)

df2['date'] = pd.to_datetime(df2['TIME'],format='%d-%m-%Y %H:%M')
df2.index = df2['date']

def generate_feature(station_id,date,n,unit):
    value = None
    new_date = date - pd.to_timedelta(n,unit=unit)
    #print('old date : {} and new date : {}'.format(date,new_date))
    data = df2[(df2['STATION ID'] == station_id)& (df2.index.date == new_date.date())& (abs((df2.index -new_date).total_seconds())<151)]
    #data = df2[(df2['STATION ID'] == station_id)& (df2['date'] == new_date)]
    if not data.empty:
        value = data['AVAILABLE BIKE STANDS'][0]    
    return value
    
def impute_values(record,column_name):
    if np.isnan(record[column_name]):
        df_filtered = df2[df2['STATION ID'] == record['STATION ID']]
        mean_value = df_filtered[column_name].mean()
        return mean_value
    else:
        return record[column_name]

print('Feature Engineering in progress (Takes about an hour)')        
df2["Day of the Month"] = df2.index.day
df2["Day of the Week"] = df2.index.weekday
df2["Hour of the Day"] = df2.index.hour
df2['hourly_1'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],1,'h'), axis = 1)
df2['hourly_1'] = df2.apply( lambda x: impute_values(x,'hourly_1'), axis = 1)
df2['hourly_2'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],2,'h'), axis = 1)
df2['hourly_2'] = df2.apply( lambda x: impute_values(x,'hourly_2'), axis = 1)
df2['hourly_3'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],3,'h'), axis = 1)
df2['hourly_3'] = df2.apply( lambda x: impute_values(x,'hourly_3'), axis = 1)
df2['daily_1'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],1,'d'), axis = 1)
df2['daily_1'] = df2.apply( lambda x: impute_values(x,'daily_1'), axis = 1)
df2['daily_2'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],2,'d'), axis = 1)
df2['daily_2'] = df2.apply( lambda x: impute_values(x,'daily_2'), axis = 1)
df2['daily_3'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],3,'d'), axis = 1)
df2['daily_3'] = df2.apply( lambda x: impute_values(x,'daily_3'), axis = 1)
df2['weekly_1'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],1,'W'), axis = 1)
df2['weekly_1'] = df2.apply( lambda x: impute_values(x,'weekly_1'), axis = 1)
df2['weekly_2'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],2,'W'), axis = 1)
df2['weekly_2'] = df2.apply( lambda x: impute_values(x,'weekly_2'), axis = 1)
df2['weekly_3'] = df2.apply( lambda x: generate_feature(x['STATION ID'],x['date'],3,'W'), axis = 1)
df2['weekly_3'] = df2.apply( lambda x: impute_values(x,'weekly_3'), axis = 1)
df2.drop(['STATION ID','TIME', 'date' , ], axis=1,inplace=True)

df2['date'] = df2.index
df2.index = indices_original
df_to_normalize=df2.iloc[:,3:15]
df_normalized = normalize(df_to_normalize,norm='max', axis=0)
df_normalized_df= pd.DataFrame(df_normalized,index= indices_original)
df_rest=df2.iloc[:,0:3]
df_new = pd.concat([df_rest, df_normalized_df], axis=1,join='inner')
df_new.index = df2['date']
df_new.columns= [ 'AVAILABLE BIKE STANDS', 'STATION_33', 'STATION_107',
       'Day of the Month', 'Day of the Week', 'Hour of the Day', 'hourly_1',
       'hourly_2', 'hourly_3', 'daily_1', 'daily_2', 'daily_3', 'weekly_1',
       'weekly_2', 'weekly_3']
df2 = df_new
df3 = df2.dropna()
print('Feature Engineering completed')  

X = df3.iloc[:, 1:]
y=df3.loc[:, 'AVAILABLE BIKE STANDS'] 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.10, random_state=42)

def naive_baseline_model(series,station_id):
    values = pd.DataFrame(series.values)
    dataframe = pd.concat([values.shift(12), values], axis=1)
    dataframe.columns = ['t-1', 't+1']

    # split into train and test sets
    X = dataframe.values
    train_size = int(len(X) * 0.66)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]
    test_X, test_y = test[:,0], test[:,1]

    predictions = list()
    for x in test_X:
        yhat = x
        predictions.append(yhat)
    test_score = mean_squared_error(test_y, predictions)
    return test_score
    
def naive_baseline(series_1,series_2):
    mean_error = (naive_baseline_model(series_1,33) + naive_baseline_model(series_2,107))/2
    print('Mean squared error for baseline model: {}'.format(mean_error))

series_1 = y[X['STATION_33']==1]
series_2 = y[X['STATION_107']==1]
naive_baseline(series_1,series_2)

print("Lasso Linear Regression")
mean_err=[]
std_err=[]
scores_mean=[]
scores_std=[]
C=[0.0001,0.001,0.01]
for c in C:
    model = linear_model.Lasso(alpha=1/(2*c))
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    mse = mean_squared_error(y_test, ypred)
    mean_err.append(np.array(mse).mean())
    std_err.append(np.array(mse).std())
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    #print('Cross Validation Scores Mean: {}'.format(np.array(scores).mean()))
    scores_mean.append(np.array(scores).mean())
    scores_std.append(np.array(scores).std())
    #print('Cross Validation Scores Standard Deviation: {}'.format(np.array(scores).std()))
plot1 = plt.figure(1)
plt.errorbar( C,mean_err, yerr=std_err)
plt.xlabel('C') 
plt.ylabel('Mean Square Error')
plot2 = plt.figure(2)
plt.errorbar( C,scores_mean, yerr=scores_std)
plt.xlabel('C') 
plt.ylabel('Cross validation Error')
plt.show()

print("Ridge Linear Regression")
mean_err=[]
std_err=[]
scores_mean=[]
scores_std=[]
C=[0.0001,0.001,0.01,0.1,1]
for c in C:
    model = linear_model.Ridge(alpha=1/(2*c))
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    mse = mean_squared_error(y_test, ypred)
    mean_err.append(np.array(mse).mean())
    std_err.append(np.array(mse).std())
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    #print('Cross Validation Scores Mean: {}'.format(np.array(scores).mean()))
    scores_mean.append(np.array(scores).mean()*-1)
    scores_std.append(np.array(scores).std())
    #print('Cross Validation Scores Standard Deviation: {}'.format(np.array(scores).std()))
plot1 = plt.figure(1)
plt.errorbar( C,mean_err, yerr=std_err)
plt.xlabel('C') 
plt.ylabel('Mean Square Error')
plot2 = plt.figure(2)
plt.errorbar( C,scores_mean, yerr=scores_std)
plt.xlabel('C') 
plt.ylabel('Cross validation Error')
plt.show()

c=0.001
linear_reg_model = linear_model.Ridge(alpha=1/(2*c)).fit( X_train, y_train)
y_pred = linear_reg_model.predict(X_test)
y_train_pred = linear_reg_model.predict(X_train)
print("Mean squared error of test data on Ridge Regression is %f"%(mean_squared_error(y_test,y_pred)))
print("Mean squared error of train data on Ridge Regression is %f"%(mean_squared_error(y_train,y_train_pred)))

DayOfMonth = np.sort(X_test['Day of the Month'].unique())
DayOfWeek = np.sort(X_test['Day of the Week'].unique())
HourOfDay = np.sort(X_test['Hour of the Day'].unique())

# Plot of train predictions station 33
print('Plot of train predictions station 33')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_train[X_train['STATION_33']==1]
df_plot_pred['Occupancy'] = y_train_pred[X_train['STATION_33']==1]
df_plot.index = y_train[X_train['STATION_33']==1].index
df_plot_pred.index = y_train[X_train['STATION_33']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

# Plot of train predictions station 107
print('Plot of train predictions station 107')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_train[X_train['STATION_107']==1]
df_plot_pred['Occupancy'] = y_train_pred[X_train['STATION_107']==1]
df_plot.index = y_train[X_train['STATION_107']==1].index
df_plot_pred.index = y_train[X_train['STATION_107']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

# Plot of test predictions station 33
print('Plot of test predictions station 33')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_test[X_test['STATION_33']==1]
df_plot_pred['Occupancy'] = y_pred[X_test['STATION_33']==1]
df_plot.index = y_test[X_test['STATION_33']==1].index
df_plot_pred.index = y_test[X_test['STATION_33']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

# Plot of test predictions station 107
print('Plot of test predictions station 107')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_test[X_test['STATION_107']==1]
df_plot_pred['Occupancy'] = y_pred[X_test['STATION_107']==1]
df_plot.index = y_test[X_test['STATION_107']==1].index
df_plot_pred.index = y_test[X_test['STATION_107']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

df2.index = pd.to_datetime(df2.index,format='%Y-%m-%d %H:%M')

def gen_Value(station_id,date,n,unit,column_name):
    value = None
    new_date = date - pd.to_timedelta(n,unit=unit)
    station_name = 'STATION_'+str(station_id)
    #print('old date : {} and new date : {}'.format(date,new_date))
    if station_id ==107:
        data = df2[(df2['STATION_107'] == 1)& (df2.index.date == new_date.date())& (abs((df2.index -new_date).total_seconds())<151)]
    elif station_id == 33:
        data = df2[(df2['STATION_33'] == 1)& (df2.index.date == new_date.date())& (abs((df2.index -new_date).total_seconds())<151)]
    if data.empty:
        pass
    else:
        value = data[column_name][0]
    if value == None:
        df_filtered = df2[df2[station_name] == 1]
        mean_value = df_filtered[column_name].mean()
        value = mean_value
    return value

def pred_future(date,station_id):
    df_new = pd.DataFrame(data= [[0,0,0,0,0,0,0,0,0,0,0,0,0,0]],columns=['STATION_33', 'STATION_107','Day of the Month','Day of the Week','Hour of the Day', 'hourly_1', 'hourly_2', 'hourly_3',
       'daily_1', 'daily_2', 'daily_3', 'weekly_1', 'weekly_2', 'weekly_3'])
    time = pd.to_datetime(date,format='%Y-%m-%d %H:%M')
    time.replace(second=0)
    
    if station_id == 33:
        df_new['STATION_33'] = 1
    elif station_id == 107 :
        df_new['STATION_107'] = 1

   
        
    # 10 minute future
    new_time = time + pd.to_timedelta(10,unit='minutes')
    #df_new['Weekday'] = generate_weekdays(new_time)
    #df_new['Weekend'] = generate_weekend(new_time)
    df_new["Day of the Month"] = DayOfMonth[new_time.day]
    df_new["Day of the Week"] = DayOfWeek[new_time.weekday()]
    df_new["Hour of the Day"] = HourOfDay[new_time.hour]
    df_new['hourly_1'] = gen_Value(station_id,new_time,1,'h','hourly_1')
    df_new['hourly_2'] = gen_Value(station_id,new_time,2,'h','hourly_2')
    df_new['hourly_3'] = gen_Value(station_id,new_time,3,'h','hourly_3')
    df_new['daily_1'] = gen_Value(station_id,new_time,1,'d','daily_1')
    df_new['daily_2'] = gen_Value(station_id,new_time,2,'d','daily_2')
    df_new['daily_3'] = gen_Value(station_id,new_time,3,'d','daily_3')
    df_new['weekly_1'] = gen_Value(station_id,new_time,1,'W','weekly_1')
    df_new['weekly_2'] = gen_Value(station_id,new_time,2,'W','weekly_2')
    df_new['weekly_3'] = gen_Value(station_id,new_time,3,'W','weekly_3')
    bikes_available_in_10minutes = linear_reg_model.predict(df_new)
    
    # 30 minute future
    new_time = time + pd.to_timedelta(30,unit='minutes')
    #df_new['Weekday'] = generate_weekdays(new_time)
    #df_new['Weekend'] = generate_weekend(new_time)
    df_new["Day of the Month"] = DayOfMonth[new_time.day]
    df_new["Day of the Week"] = DayOfWeek[new_time.weekday()]
    df_new["Hour of the Day"] = HourOfDay[new_time.hour]
    df_new['hourly_1'] = gen_Value(station_id,new_time,1,'h','hourly_1')
    df_new['hourly_2'] = gen_Value(station_id,new_time,2,'h','hourly_2')
    df_new['hourly_3'] = gen_Value(station_id,new_time,3,'h','hourly_3')
    df_new['daily_1'] = gen_Value(station_id,new_time,1,'d','daily_1')
    df_new['daily_2'] = gen_Value(station_id,new_time,2,'d','daily_2')
    df_new['daily_3'] = gen_Value(station_id,new_time,3,'d','daily_3')
    df_new['weekly_1'] = gen_Value(station_id,new_time,1,'W','weekly_1')
    df_new['weekly_2'] = gen_Value(station_id,new_time,2,'W','weekly_2')
    df_new['weekly_3'] = gen_Value(station_id,new_time,3,'W','weekly_3')
    bikes_available_in_30minutes = linear_reg_model.predict(df_new)
    
    # 1 hour future
    new_time = time + pd.to_timedelta(1,unit='h')
    #df_new['Weekday'] = generate_weekdays(new_time)
    #df_new['Weekend'] = generate_weekend(new_time)
    df_new["Day of the Month"] = DayOfMonth[new_time.day]
    df_new["Day of the Week"] = DayOfWeek[new_time.weekday()]
    df_new["Hour of the Day"] = HourOfDay[new_time.hour]
    df_new['hourly_1'] = gen_Value(station_id,new_time,1,'h','hourly_1')
    df_new['hourly_2'] = gen_Value(station_id,new_time,2,'h','hourly_2')
    df_new['hourly_3'] = gen_Value(station_id,new_time,3,'h','hourly_3')
    df_new['daily_1'] = gen_Value(station_id,new_time,1,'d','daily_1')
    df_new['daily_2'] = gen_Value(station_id,new_time,2,'d','daily_2')
    df_new['daily_3'] = gen_Value(station_id,new_time,3,'d','daily_3')
    df_new['weekly_1'] = gen_Value(station_id,new_time,1,'W','weekly_1')
    df_new['weekly_2'] = gen_Value(station_id,new_time,2,'W','weekly_2')
    df_new['weekly_3'] = gen_Value(station_id,new_time,3,'W','weekly_3')
    bikes_available_in_1hour = linear_reg_model.predict(df_new)
    
    return round(bikes_available_in_10minutes[0]),round(bikes_available_in_30minutes[0]),round(bikes_available_in_1hour[0])
    
station = 33
time = '2020-03-01 10:50:02'
x,y,z = pred_future(time,station)
print('Linear Regression Prediction:')
print(f"The occupancy of the station {station} for time: {time}  at 10 minutes future is {x}, 30 minutes future is {y} and 1 hour future is {z}.")

print("Random Forest Regression")
mean_err=[]
std_err=[]
scores_mean=[]
scores_std=[]
max_depth=[10,15,17,20,25,27,30]
for m in max_depth:
    model = RandomForestRegressor(max_depth=m, random_state=0, min_impurity_decrease= 0.008) 
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    mse = mean_squared_error(y_test, ypred)
    mean_err.append(np.array(mse).mean())
    std_err.append(np.array(mse).std())
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    scores_mean.append(np.array(scores).mean()*-1)
    scores_std.append(np.array(scores).std())
plot1 = plt.figure(1)
plt.errorbar( max_depth,mean_err, yerr=std_err)
plt.xlabel('Maximum Depth') 
plt.ylabel('Mean Square Error')
plot2 = plt.figure(2)
plt.errorbar( max_depth,scores_mean, yerr=scores_std)
plt.xlabel('Maximum Depth') 
plt.ylabel('Cross validation Error')
plt.show()

regr = RandomForestRegressor(max_depth=15, random_state=0, min_impurity_decrease= 0.008) #, min_impurity_decrease= 0.005, max_features = 10 ) 
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
y_train_pred = regr.predict(X_train)
print("Mean squared error of test data on Random Forest Regression is  %f"%(mean_squared_error(y_test,y_pred)))
print("Mean squared error of train data on Random Forest Regression is  %f"%(mean_squared_error(y_train,y_train_pred)))
 
# Plot of train predictions station 33
print('Plot of train predictions station 33')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_train[X_train['STATION_33']==1]
df_plot_pred['Occupancy'] = y_train_pred[X_train['STATION_33']==1]
df_plot.index = y_train[X_train['STATION_33']==1].index
df_plot_pred.index = y_train[X_train['STATION_33']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

# Plot of train predictions station 107
print('Plot of train predictions station 107')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_train[X_train['STATION_107']==1]
df_plot_pred['Occupancy'] = y_train_pred[X_train['STATION_107']==1]
df_plot.index = y_train[X_train['STATION_107']==1].index
df_plot_pred.index = y_train[X_train['STATION_107']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

# Plot of test predictions station 33
print('Plot of test predictions station 33')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_test[X_test['STATION_33']==1]
df_plot_pred['Occupancy'] = y_pred[X_test['STATION_33']==1]
df_plot.index = y_test[X_test['STATION_33']==1].index
df_plot_pred.index = y_test[X_test['STATION_33']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

# Plot of test predictions station 107
print('Plot of test predictions station 107')
df_plot = pd.DataFrame()
df_plot_pred = pd.DataFrame()
df_plot['Occupancy'] = y_test[X_test['STATION_107']==1]
df_plot_pred['Occupancy'] = y_pred[X_test['STATION_107']==1]
df_plot.index = y_test[X_test['STATION_107']==1].index
df_plot_pred.index = y_test[X_test['STATION_107']==1].index
fig, ax = plt.subplots()
df_plot.plot(c='blue', ax = ax)
df_plot_pred.plot(c='red', ax = ax)
ax.legend( ['Occupancy Actual', 'Occupancy Predicted'])
plt.show()

def pred_future_rf(date,station_id):
    df_new = pd.DataFrame(data= [[0,0,0,0,0,0,0,0,0,0,0,0,0,0]],columns=['STATION_33', 'STATION_107','Day of the Month','Day of the Week','Hour of the Day', 'hourly_1', 'hourly_2', 'hourly_3',
       'daily_1', 'daily_2', 'daily_3', 'weekly_1', 'weekly_2', 'weekly_3'])
    time = pd.to_datetime(date,format='%Y-%m-%d %H:%M')
    time.replace(second=0)
    
    if station_id == 33:
        df_new['STATION_33'] = 1
    elif station_id == 107 :
        df_new['STATION_107'] = 1

        
        
    # 10 minute future
    new_time = time + pd.to_timedelta(10,unit='minutes')
    #df_new['Weekday'] = generate_weekdays(new_time)
    #df_new['Weekend'] = generate_weekend(new_time)
    df_new["Day of the Month"] = DayOfMonth[new_time.day]
    df_new["Day of the Week"] = DayOfWeek[new_time.weekday()]
    df_new["Hour of the Day"] = HourOfDay[new_time.hour]
    df_new['hourly_1'] = gen_Value(station_id,new_time,1,'h','hourly_1')
    df_new['hourly_2'] = gen_Value(station_id,new_time,2,'h','hourly_2')
    df_new['hourly_3'] = gen_Value(station_id,new_time,3,'h','hourly_3')
    df_new['daily_1'] = gen_Value(station_id,new_time,1,'d','daily_1')
    df_new['daily_2'] = gen_Value(station_id,new_time,2,'d','daily_2')
    df_new['daily_3'] = gen_Value(station_id,new_time,3,'d','daily_3')
    df_new['weekly_1'] = gen_Value(station_id,new_time,1,'W','weekly_1')
    df_new['weekly_2'] = gen_Value(station_id,new_time,2,'W','weekly_2')
    df_new['weekly_3'] = gen_Value(station_id,new_time,3,'W','weekly_3')
    bikes_available_in_10minutes = regr.predict(df_new)
    
    # 30 minute future
    new_time = time + pd.to_timedelta(30,unit='minutes')
    #df_new['Weekday'] = generate_weekdays(new_time)
    #df_new['Weekend'] = generate_weekend(new_time)
    df_new["Day of the Month"] = DayOfMonth[new_time.day]
    df_new["Day of the Week"] = DayOfWeek[new_time.weekday()]
    df_new["Hour of the Day"] = HourOfDay[new_time.hour]
    df_new['hourly_1'] = gen_Value(station_id,new_time,1,'h','hourly_1')
    df_new['hourly_2'] = gen_Value(station_id,new_time,2,'h','hourly_2')
    df_new['hourly_3'] = gen_Value(station_id,new_time,3,'h','hourly_3')
    df_new['daily_1'] = gen_Value(station_id,new_time,1,'d','daily_1')
    df_new['daily_2'] = gen_Value(station_id,new_time,2,'d','daily_2')
    df_new['daily_3'] = gen_Value(station_id,new_time,3,'d','daily_3')
    df_new['weekly_1'] = gen_Value(station_id,new_time,1,'W','weekly_1')
    df_new['weekly_2'] = gen_Value(station_id,new_time,2,'W','weekly_2')
    df_new['weekly_3'] = gen_Value(station_id,new_time,3,'W','weekly_3')
    bikes_available_in_30minutes = regr.predict(df_new)
    
    # 1 hour future
    new_time = time + pd.to_timedelta(1,unit='h')
    #df_new['Weekday'] = generate_weekdays(new_time)
    #df_new['Weekend'] = generate_weekend(new_time)
    df_new["Day of the Month"] = DayOfMonth[new_time.day]
    df_new["Day of the Week"] = DayOfWeek[new_time.weekday()]
    df_new["Hour of the Day"] = HourOfDay[new_time.hour]
    df_new['hourly_1'] = gen_Value(station_id,new_time,1,'h','hourly_1')
    df_new['hourly_2'] = gen_Value(station_id,new_time,2,'h','hourly_2')
    df_new['hourly_3'] = gen_Value(station_id,new_time,3,'h','hourly_3')
    df_new['daily_1'] = gen_Value(station_id,new_time,1,'d','daily_1')
    df_new['daily_2'] = gen_Value(station_id,new_time,2,'d','daily_2')
    df_new['daily_3'] = gen_Value(station_id,new_time,3,'d','daily_3')
    df_new['weekly_1'] = gen_Value(station_id,new_time,1,'W','weekly_1')
    df_new['weekly_2'] = gen_Value(station_id,new_time,2,'W','weekly_2')
    df_new['weekly_3'] = gen_Value(station_id,new_time,3,'W','weekly_3')
    bikes_available_in_1hour = regr.predict(df_new)
    
    return round(bikes_available_in_10minutes[0]),round(bikes_available_in_30minutes[0]),round(bikes_available_in_1hour[0])
    
station = 33
time = '2020-03-01 10:50:02'
x,y,z = pred_future_rf(time,station)
print('Random Forest Regression Prediction:')
print(f"The occupancy of the station {station} for time: {time} at 10 minutes future is {x}, 30 minutes future is {y} and 1 future is {z}.")