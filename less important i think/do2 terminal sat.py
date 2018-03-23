jl@jacob-L430:~$ xrandr
Screen 0: minimum 320 x 200, current 3286 x 1200, maximum 8192 x 8192
LVDS-1 connected primary 1366x768+0+0 (normal left inverted right x axis y axis) 310mm x 174mm
   1366x768      59.98*+
   1360x768      59.80    59.96
   1024x768      60.04    60.00
   960x720       60.00
   928x696       60.05
   896x672       60.01
   960x600       60.00
   960x540       59.99
   800x600       60.00    60.32    56.25
   840x525       60.01    59.88
   800x512       60.17
   700x525       59.98
   640x512       60.02
   720x450       59.89
   640x480       60.00    59.94
   680x384       59.80    59.96
   576x432       60.06
   512x384       60.00
   400x300       60.32    56.34
   320x240       60.05
VGA-1 disconnected (normal left inverted right x axis y axis)
HDMI-1 disconnected (normal left inverted right x axis y axis)
DP-1 connected 1920x1200+1366+0 (normal left inverted right x axis y axis) 518mm x 324mm
   1920x1200     59.95*+
   1920x1080     60.00
   1600x1200     60.00
   1680x1050     59.95
   1280x1024     60.02
   1280x960      60.00
   1024x768      60.00
   800x600       60.32
   640x480       59.94
   720x400       70.08
HDMI-2 disconnected (normal left inverted right x axis y axis)
HDMI-3 disconnected (normal left inverted right x axis y axis)
DP-2 disconnected (normal left inverted right x axis y axis)
DP-3 disconnected (normal left inverted right x axis y axis)
jl@jacob-L430:~$ xrandr --output LVDS-1 --off
jl@jacob-L430:~$ cd Dropbox/aap/JLPM/defaults/
jl@jacob-L430:~/Dropbox/aap/JLPM/defaults$ python
Python 3.6.3 |Anaconda custom (64-bit)| (default, Oct 13 2017, 12:02:49)
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import matplotlib.pyplot as plt
>>> import pandas as pd
>>> from sklearn.cross_validation import train_test_split
/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
>>> import numpy as np
>>> from numpy import*
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn import metrics
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.cross_validation import cross_val_score
>>>
>>> ## Import data
...
>>> raw_data = pd.read_csv('dataset.csv', sep=";")
>>>
>>>
>>> df_test = raw_data[pd.isnull(raw_data.default)]
>>> # df_modeling contains all rows that have deafult=1 or 0
... df_modeling = raw_data[pd.notnull(raw_data.default)]
>>> # cehck fraction of defaults
... df_modeling['default'].mean() #concl: 0.014 so very few deafults
0.014314928425357873
>>> # df_predicting contains all rows that have deafult=NA
... df_predicting = raw_data[pd.isnull(raw_data.default)]
>>> # df_modeling contains all rows that have deafult=1 or 0
... df_modeling = raw_data[pd.notnull(raw_data.default)]
>>> # cehck fraction of defaults
... df_modeling['default'].mean() #concl: 0.014 so very few deafults
0.014314928425357873
>>> y_predicting.heda()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_predicting' is not defined
>>> y_predicting.head()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_predicting' is not defined
>>> df_predicting.head()
                                       uuid  default  \
89976  6f6e6c6a-2081-4e6b-8eb3-4fd89b54b2d7      NaN
89977  f6f6d9f3-ef2b-4329-a388-c6a687f27e70      NaN
89978  e9c39869-1bc5-4375-b627-a2df70b445ea      NaN
89979  6beb88a3-9641-4381-beb6-c9a208664dd0      NaN
89980  bb89b735-72fe-42a4-ba06-d63be0f4ca36      NaN

       account_amount_added_12_24m  account_days_in_dc_12_24m  \
89976                            0                        0.0
89977                            0                        0.0
89978                        50956                        0.0
89979                        35054                        0.0
89980                            0                        0.0

       account_days_in_rem_12_24m  account_days_in_term_12_24m  \
89976                         0.0                          0.0
89977                         0.0                          0.0
89978                        77.0                          0.0
89979                         0.0                          0.0
89980                         0.0                          0.0

       account_incoming_debt_vs_paid_0_24m  account_status  \
89976                             0.009135             1.0
89977                                  NaN             NaN
89978                             0.000000             1.0
89979                             0.000000             1.0
89980                             0.000000             1.0

       account_worst_status_0_3m  account_worst_status_12_24m  \
89976                        1.0                          NaN
89977                        NaN                          NaN
89978                        1.0                          2.0
89979                        1.0                          1.0
89980                        2.0                          NaN

                ...             status_3rd_last_archived_0_24m  \
89976           ...                                          1
89977           ...                                          0
89978           ...                                          2
89979           ...                                          0
89980           ...                                          0

       status_max_archived_0_6_months  status_max_archived_0_12_months  \
89976                               1                                1
89977                               0                                0
89978                               1                                1
89979                               2                                2
89980                               0                                0

       status_max_archived_0_24_months  recovery_debt  \
89976                                1              0
89977                                0              0
89978                                3              0
89979                                2              0
89980                                0              0

      sum_capital_paid_account_0_12m sum_capital_paid_account_12_24m  \
89976                           8815                               0
89977                              0                               0
89978                          36163                           39846
89979                          62585                               0
89980                          14295                               0

       sum_paid_inv_0_12m  time_hours  worst_status_active_inv
89976               27157   19.895556                      NaN
89977                   0    0.236667                      NaN
89978               93760   20.332778                      NaN
89979                1790    6.201111                      NaN
89980                   0    8.451111                      NaN

[5 rows x 43 columns]
>>> df_predicting.default.head()
89976   NaN
89977   NaN
89978   NaN
89979   NaN
89980   NaN
Name: default, dtype: float64
>>> df_predicting.default.tail()
99971   NaN
99972   NaN
99973   NaN
99974   NaN
99975   NaN
Name: default, dtype: float64
>>> df_predicting.default.isna().sum()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/pandas/core/generic.py", line 3081, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'Series' object has no attribute 'isna'
>>> df_predicting.default.isnul().sum()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/pandas/core/generic.py", line 3081, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'Series' object has no attribute 'isnul'
>>> df_predicting.default.isnull().sum()
10000
>>> df_predicting.default.notnull().sum()
0
>>> 0 == df_modeling.default.isnull().sum() #True
True
>>> df_modeling.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 89976 entries, 0 to 89975
Data columns (total 43 columns):
uuid                                   89976 non-null object
default                                89976 non-null float64
account_amount_added_12_24m            89976 non-null int64
account_days_in_dc_12_24m              79293 non-null float64
account_days_in_rem_12_24m             79293 non-null float64
account_days_in_term_12_24m            79293 non-null float64
account_incoming_debt_vs_paid_0_24m    36619 non-null float64
account_status                         41042 non-null float64
account_worst_status_0_3m              41042 non-null float64
account_worst_status_12_24m            29921 non-null float64
account_worst_status_3_6m              38038 non-null float64
account_worst_status_6_12m             35663 non-null float64
age                                    89976 non-null int64
avg_payment_span_0_12m                 68508 non-null float64
avg_payment_span_0_3m                  45594 non-null float64
merchant_category                      89976 non-null object
merchant_group                         89976 non-null object
has_paid                               89976 non-null bool
max_paid_inv_0_12m                     89976 non-null float64
max_paid_inv_0_24m                     89976 non-null float64
name_in_email                          89976 non-null object
num_active_div_by_paid_inv_0_12m       69318 non-null float64
num_active_inv                         89976 non-null int64
num_arch_dc_0_12m                      89976 non-null int64
num_arch_dc_12_24m                     89976 non-null int64
num_arch_ok_0_12m                      89976 non-null int64
num_arch_ok_12_24m                     89976 non-null int64
num_arch_rem_0_12m                     89976 non-null int64
num_arch_written_off_0_12m             73671 non-null float64
num_arch_written_off_12_24m            73671 non-null float64
num_unpaid_bills                       89976 non-null int64
status_last_archived_0_24m             89976 non-null int64
status_2nd_last_archived_0_24m         89976 non-null int64
status_3rd_last_archived_0_24m         89976 non-null int64
status_max_archived_0_6_months         89976 non-null int64
status_max_archived_0_12_months        89976 non-null int64
status_max_archived_0_24_months        89976 non-null int64
recovery_debt                          89976 non-null int64
sum_capital_paid_account_0_12m         89976 non-null int64
sum_capital_paid_account_12_24m        89976 non-null int64
sum_paid_inv_0_12m                     89976 non-null int64
time_hours                             89976 non-null float64
worst_status_active_inv                27436 non-null float64
dtypes: bool(1), float64(19), int64(19), object(4)
memory usage: 29.6+ MB
>>> # dfp contains all rows that have deafult=NA
... dfp = raw_data[pd.isnull(raw_data.default)]
>>> 0 == dfp.default.notnull().sum() #True
True
>>> # dfm contains all rows that have deafult=1 or 0
... dfm = raw_data[pd.notnull(raw_data.default)]
>>> 0 == dfm.default.isnull().sum() #True
True
>>> # cehck fraction of defaults
... dfm['default'].mean() #concl: 0.014 so very few deafults
0.014314928425357873
>>>
>>> numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'vardescr' is not defined
>>> dfm = dfm[numerical_variables]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'numerical_variables' is not defined
>>> vardescr = pd.read_csv('variabledescr.csv')
>>> numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
>>> dfm = dfm[numerical_variables]
>>> dfm.head()
   account_amount_added_12_24m  account_days_in_dc_12_24m  \
0                            0                        0.0
1                            0                        0.0
2                            0                        0.0
3                            0                        NaN
4                            0                        0.0

   account_days_in_rem_12_24m  account_days_in_term_12_24m  \
0                         0.0                          0.0
1                         0.0                          0.0
2                         0.0                          0.0
3                         NaN                          NaN
4                         0.0                          0.0

   account_incoming_debt_vs_paid_0_24m  age  avg_payment_span_0_12m  \
0                                  0.0   20               12.692308
1                                  NaN   50               25.833333
2                                  NaN   22               20.000000
3                                  NaN   36                4.687500
4                                  NaN   25               13.000000

   avg_payment_span_0_3m  max_paid_inv_0_12m  max_paid_inv_0_24m     ...      \
0               8.333333             31638.0             31638.0     ...
1              25.000000             13749.0             13749.0     ...
2              18.000000             29890.0             29890.0     ...
3               4.888889             40040.0             40040.0     ...
4              13.000000              7100.0              7100.0     ...

   num_arch_ok_12_24m  num_arch_rem_0_12m  num_arch_written_off_0_12m  \
0                  14                   0                         0.0
1                  19                   3                         0.0
2                   0                   3                         0.0
3                  21                   0                         0.0
4                   0                   0                         0.0

   num_arch_written_off_12_24m  num_unpaid_bills  recovery_debt  \
0                          0.0                 2              0
1                          0.0                 0              0
2                          0.0                 1              0
3                          0.0                 1              0
4                          0.0                 0              0

   sum_capital_paid_account_0_12m  sum_capital_paid_account_12_24m  \
0                               0                                0
1                               0                                0
2                               0                                0
3                               0                                0
4                               0                                0

   sum_paid_inv_0_12m  time_hours
0              178839    9.653333
1               49014   13.181389
2              124839   11.561944
3              324676   15.751111
4                7100   12.698611

[5 rows x 25 columns]
>>>
>>> dfp = raw_data[pd.isnull(raw_data.default)]
>>> 0 == dfp.default.notnull().sum() #True
True
>>> Yp = dfp['default']
>>> Xp = dfp.drop(columns='default')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: drop() got an unexpected keyword argument 'columns'
>>> Xp = dfp.drop(columns=['default'])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: drop() got an unexpected keyword argument 'columns'
>>> Xp = dfp.drop('default', axis=1)
>>> Xp.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 89976 to 99975
Data columns (total 42 columns):
uuid                                   10000 non-null object
account_amount_added_12_24m            10000 non-null int64
account_days_in_dc_12_24m              8847 non-null float64
account_days_in_rem_12_24m             8847 non-null float64
account_days_in_term_12_24m            8847 non-null float64
account_incoming_debt_vs_paid_0_24m    4042 non-null float64
account_status                         4561 non-null float64
account_worst_status_0_3m              4561 non-null float64
account_worst_status_12_24m            3294 non-null float64
account_worst_status_3_6m              4236 non-null float64
account_worst_status_6_12m             3963 non-null float64
age                                    10000 non-null int64
avg_payment_span_0_12m                 7632 non-null float64
avg_payment_span_0_3m                  5077 non-null float64
merchant_category                      10000 non-null object
merchant_group                         10000 non-null object
has_paid                               10000 non-null bool
max_paid_inv_0_12m                     10000 non-null float64
max_paid_inv_0_24m                     10000 non-null float64
name_in_email                          10000 non-null object
num_active_div_by_paid_inv_0_12m       7719 non-null float64
num_active_inv                         10000 non-null int64
num_arch_dc_0_12m                      10000 non-null int64
num_arch_dc_12_24m                     10000 non-null int64
num_arch_ok_0_12m                      10000 non-null int64
num_arch_ok_12_24m                     10000 non-null int64
num_arch_rem_0_12m                     10000 non-null int64
num_arch_written_off_0_12m             8227 non-null float64
num_arch_written_off_12_24m            8227 non-null float64
num_unpaid_bills                       10000 non-null int64
status_last_archived_0_24m             10000 non-null int64
status_2nd_last_archived_0_24m         10000 non-null int64
status_3rd_last_archived_0_24m         10000 non-null int64
status_max_archived_0_6_months         10000 non-null int64
status_max_archived_0_12_months        10000 non-null int64
status_max_archived_0_24_months        10000 non-null int64
recovery_debt                          10000 non-null int64
sum_capital_paid_account_0_12m         10000 non-null int64
sum_capital_paid_account_12_24m        10000 non-null int64
sum_paid_inv_0_12m                     10000 non-null int64
time_hours                             10000 non-null float64
worst_status_active_inv                3025 non-null float64
dtypes: bool(1), float64(18), int64(19), object(4)
memory usage: 3.2+ MB
>>>
>>> Ym.shape
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'Ym' is not defined
>>> Ym.shape()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'Ym' is not defined
>>> Ym
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'Ym' is not defined
>>> # dfp contains all rows that have deafult=NA
... dfp = raw_data[pd.isnull(raw_data.default)]
>>> 0 == dfp.default.notnull().sum() #True
True
>>> Yp = dfp['default'] # Y for prediction
>>> Xp = dfp.drop('default', axis=1) # X for prediction
>>> # dfm contains all rows that have deafult=1 or 0
... dfm = raw_data[pd.notnull(raw_data.default)]
>>> 0 == dfm.default.isnull().sum() #True
True
>>> # cehck fraction of defaults
... dfm['default'].mean() #concl: 0.014 so very few deafults
0.014314928425357873
>>>
>>> # Select Y for out model "Ym"
... Ym = dfm['default']
>>> # Select X for our model "Xm"
... # select variables with type=numeric.
... numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
>>> Xm = dfm[numerical_variables]
>>> # qq increase to also include categorical
... Xm = X_df.as_matrix() #keras wants a matrix
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
NameError: name 'X_df' is not defined
>>>
>>> Xm = Xm.as_matrix() #keras wants a matrix
>>> Xm
array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   1.78839000e+05,   9.65333333e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   4.90140000e+04,   1.31813889e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   1.24839000e+05,   1.15619444e+01],
       ...,
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   4.73060000e+04,   1.86819444e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   1.35300000e+04,   1.19644444e+01],
       [  0.00000000e+00,              nan,              nan, ...,
          0.00000000e+00,   8.41000000e+03,   1.89258333e+01]])
>>> # inside dfm we split into X_train and X_test  (70% and 30%)
... # use random_state for reproducibility
... X_train, X_test, Y_train, Y_test = train_test_split(Xm, Ym, random_state=9)
>>>
>>> # Our data is divided into: Y_train, Y_test, Yp
...
>>> # standardize numerical X
... X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
NameError: name 'preprocessing' is not defined
>>> X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'preprocessing' is not defined
>>> Xp = preprocessing.StandardScaler().fit(X_test).transform(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'preprocessing' is not defined
>>> scaler = preprocessing.StandardScaler().fit(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'preprocessing' is not defined
>>> X_train = scaler.transform(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'scaler' is not defined
>>> X_test = scaler.transform(X_test
...
...
KeyboardInterrupt
>>> >>> from sklearn.model_selection import train_test_split
  File "<stdin>", line 1
    >>> from sklearn.model_selection import train_test_split
     ^
SyntaxError: invalid syntax
>>> >>> from sklearn.metrics import accuracy_score
  File "<stdin>", line 1
    >>> from sklearn.metrics import accuracy_score
     ^
SyntaxError: invalid syntax
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import accuracy_score
>>>
>>> scaler = preprocessing.StandardScaler().fit(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'preprocessing' is not defined
>>> X_train = scaler.transform(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'scaler' is not defined
>>> X_test = scaler.transform(X_test
...
...
KeyboardInterrupt
>>> X_train, X_test, Y_train, Y_test = train_test_split(Xm, Ym, random_state=9)
>>> X_train.shape
(67482, 25)
>>> X_test.shape
(22494, 25)
>>>
>>> scaler = preprocessing.StandardScaler().fit(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'preprocessing' is not defined
>>> X_train = scaler.transform(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'scaler' is not defined
>>>
>>> scaler = preprocessing.StandardScaler().fit(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'preprocessing' is not defined
>>> scaler = StandardScaler().fit(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'StandardScaler' is not defined
>>> from sklearn import neighbors, preprocessing
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import accuracy_score
>>> from sklearn.preprocessing import StandardScaler
>>>
>>> scaler = StandardScaler().fit(X_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py", line 590, in fit
    return self.partial_fit(X, y)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py", line 612, in partial_fit
    warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 453, in check_array
    _assert_all_finite(array)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 44, in _assert_all_finite
    " or a value too large for %r." % X.dtype)
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
>>> X_train
array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   0.00000000e+00,   8.51833333e+00],
       [  0.00000000e+00,              nan,              nan, ...,
          0.00000000e+00,   4.09750000e+04,   9.24527778e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   6.59000000e+03,   1.59636111e+01],
       ...,
       [  2.13100000e+04,   0.00000000e+00,   0.00000000e+00, ...,
          2.12790000e+04,   0.00000000e+00,   1.07905556e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   3.06950000e+04,   2.12527778e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   1.90350000e+04,   1.91461111e+01]])
>>> dfm[numerical_variables].info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 89976 entries, 0 to 89975
Data columns (total 25 columns):
account_amount_added_12_24m            89976 non-null int64
account_days_in_dc_12_24m              79293 non-null float64
account_days_in_rem_12_24m             79293 non-null float64
account_days_in_term_12_24m            79293 non-null float64
account_incoming_debt_vs_paid_0_24m    36619 non-null float64
age                                    89976 non-null int64
avg_payment_span_0_12m                 68508 non-null float64
avg_payment_span_0_3m                  45594 non-null float64
max_paid_inv_0_12m                     89976 non-null float64
max_paid_inv_0_24m                     89976 non-null float64
num_active_div_by_paid_inv_0_12m       69318 non-null float64
num_active_inv                         89976 non-null int64
num_arch_dc_0_12m                      89976 non-null int64
num_arch_dc_12_24m                     89976 non-null int64
num_arch_ok_0_12m                      89976 non-null int64
num_arch_ok_12_24m                     89976 non-null int64
num_arch_rem_0_12m                     89976 non-null int64
num_arch_written_off_0_12m             73671 non-null float64
num_arch_written_off_12_24m            73671 non-null float64
num_unpaid_bills                       89976 non-null int64
recovery_debt                          89976 non-null int64
sum_capital_paid_account_0_12m         89976 non-null int64
sum_capital_paid_account_12_24m        89976 non-null int64
sum_paid_inv_0_12m                     89976 non-null int64
time_hours                             89976 non-null float64
dtypes: float64(12), int64(13)
memory usage: 20.3 MB
>>> dfm[numerical_variables].isnull().sum()
account_amount_added_12_24m                0
account_days_in_dc_12_24m              10683
account_days_in_rem_12_24m             10683
account_days_in_term_12_24m            10683
account_incoming_debt_vs_paid_0_24m    53357
age                                        0
avg_payment_span_0_12m                 21468
avg_payment_span_0_3m                  44382
max_paid_inv_0_12m                         0
max_paid_inv_0_24m                         0
num_active_div_by_paid_inv_0_12m       20658
num_active_inv                             0
num_arch_dc_0_12m                          0
num_arch_dc_12_24m                         0
num_arch_ok_0_12m                          0
num_arch_ok_12_24m                         0
num_arch_rem_0_12m                         0
num_arch_written_off_0_12m             16305
num_arch_written_off_12_24m            16305
num_unpaid_bills                           0
recovery_debt                              0
sum_capital_paid_account_0_12m             0
sum_capital_paid_account_12_24m            0
sum_paid_inv_0_12m                         0
time_hours                                 0
dtype: int64
>>> Xm.dropna(inflace=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'dropna'
>>> Xm = dfm[numerical_variables]
>>> # Remove alla NA cols
... Xm.dropna(inflace=True)
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
TypeError: dropna() got an unexpected keyword argument 'inflace'
>>> Xm = dfm[numerical_variables]
>>> # Remove alla NA cols
... Xm.dropna(inplace=True)
__main__:2: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
>>> Xm = dfm[numerical_variables]
>>> # Remove alla NA cols
... Xm = Xm.dropna()
>>> Xm.isnull().sum()
account_amount_added_12_24m            0
account_days_in_dc_12_24m              0
account_days_in_rem_12_24m             0
account_days_in_term_12_24m            0
account_incoming_debt_vs_paid_0_24m    0
age                                    0
avg_payment_span_0_12m                 0
avg_payment_span_0_3m                  0
max_paid_inv_0_12m                     0
max_paid_inv_0_24m                     0
num_active_div_by_paid_inv_0_12m       0
num_active_inv                         0
num_arch_dc_0_12m                      0
num_arch_dc_12_24m                     0
num_arch_ok_0_12m                      0
num_arch_ok_12_24m                     0
num_arch_rem_0_12m                     0
num_arch_written_off_0_12m             0
num_arch_written_off_12_24m            0
num_unpaid_bills                       0
recovery_debt                          0
sum_capital_paid_account_0_12m         0
sum_capital_paid_account_12_24m        0
sum_paid_inv_0_12m                     0
time_hours                             0
dtype: int64
>>> 0 == dfm.default.isnull().sum() #True
True
>>> dfm.isnull().sum()
uuid                                       0
default                                    0
account_amount_added_12_24m                0
account_days_in_dc_12_24m              10683
account_days_in_rem_12_24m             10683
account_days_in_term_12_24m            10683
account_incoming_debt_vs_paid_0_24m    53357
account_status                         48934
account_worst_status_0_3m              48934
account_worst_status_12_24m            60055
account_worst_status_3_6m              51938
account_worst_status_6_12m             54313
age                                        0
avg_payment_span_0_12m                 21468
avg_payment_span_0_3m                  44382
merchant_category                          0
merchant_group                             0
has_paid                                   0
max_paid_inv_0_12m                         0
max_paid_inv_0_24m                         0
name_in_email                              0
num_active_div_by_paid_inv_0_12m       20658
num_active_inv                             0
num_arch_dc_0_12m                          0
num_arch_dc_12_24m                         0
num_arch_ok_0_12m                          0
num_arch_ok_12_24m                         0
num_arch_rem_0_12m                         0
num_arch_written_off_0_12m             16305
num_arch_written_off_12_24m            16305
num_unpaid_bills                           0
status_last_archived_0_24m                 0
status_2nd_last_archived_0_24m             0
status_3rd_last_archived_0_24m             0
status_max_archived_0_6_months             0
status_max_archived_0_12_months            0
status_max_archived_0_24_months            0
recovery_debt                              0
sum_capital_paid_account_0_12m             0
sum_capital_paid_account_12_24m            0
sum_paid_inv_0_12m                         0
time_hours                                 0
worst_status_active_inv                62540
dtype: int64
>>> dfm.isnull().sum().sum() > 0 #true
True
>>> dfm = dfm.dropna()
>>> dfm.isnull().sum().sum() = 0 #true
  File "<stdin>", line 1
SyntaxError: can't assign to function call
>>> dfm.isnull().sum().sum() == 0 #true
True
>>> # Select Y for out model "Ym"
... Ym = dfm['default']
>>> # Select X for our model "Xm"
... # select variables with type=numeric.
... numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
>>> Xm = dfm[numerical_variables]
>>> Xm.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 9111 entries, 7 to 89966
Data columns (total 25 columns):
account_amount_added_12_24m            9111 non-null int64
account_days_in_dc_12_24m              9111 non-null float64
account_days_in_rem_12_24m             9111 non-null float64
account_days_in_term_12_24m            9111 non-null float64
account_incoming_debt_vs_paid_0_24m    9111 non-null float64
age                                    9111 non-null int64
avg_payment_span_0_12m                 9111 non-null float64
avg_payment_span_0_3m                  9111 non-null float64
max_paid_inv_0_12m                     9111 non-null float64
max_paid_inv_0_24m                     9111 non-null float64
num_active_div_by_paid_inv_0_12m       9111 non-null float64
num_active_inv                         9111 non-null int64
num_arch_dc_0_12m                      9111 non-null int64
num_arch_dc_12_24m                     9111 non-null int64
num_arch_ok_0_12m                      9111 non-null int64
num_arch_ok_12_24m                     9111 non-null int64
num_arch_rem_0_12m                     9111 non-null int64
num_arch_written_off_0_12m             9111 non-null float64
num_arch_written_off_12_24m            9111 non-null float64
num_unpaid_bills                       9111 non-null int64
recovery_debt                          9111 non-null int64
sum_capital_paid_account_0_12m         9111 non-null int64
sum_capital_paid_account_12_24m        9111 non-null int64
sum_paid_inv_0_12m                     9111 non-null int64
time_hours                             9111 non-null float64
dtypes: float64(12), int64(13)
memory usage: 1.8 MB
>>> Xm = Xm.as_matrix() #keras wants a matrix
>>> Ym
7        0.0
8        0.0
47       0.0
50       0.0
53       0.0
54       0.0
72       0.0
102      0.0
106      0.0
110      0.0
116      0.0
117      0.0
133      0.0
138      0.0
158      0.0
167      0.0
169      0.0
179      0.0
181      0.0
183      0.0
217      0.0
237      0.0
246      0.0
269      0.0
273      0.0
274      0.0
277      0.0
287      0.0
298      0.0
299      0.0
        ...
89618    0.0
89632    0.0
89647    0.0
89651    0.0
89661    0.0
89664    0.0
89674    0.0
89677    0.0
89682    0.0
89690    0.0
89692    0.0
89718    0.0
89730    0.0
89738    0.0
89760    0.0
89782    0.0
89786    0.0
89831    0.0
89860    0.0
89886    0.0
89891    0.0
89893    0.0
89915    0.0
89916    0.0
89920    0.0
89921    1.0
89928    0.0
89939    0.0
89940    0.0
89966    0.0
Name: default, Length: 9111, dtype: float64
>>> Ym.shape()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object is not callable
>>> Ym.shape
(9111,)
>>> Xm.shape
(9111, 25)
>>> Ym.shape[0] == Xm.shape[0] #true
True
>>> X_train, X_test, Y_train, Y_test = train_test_split(Xm, Ym, random_state=9)
>>> (X_train.shape, X_test.shape)
((6833, 25), (2278, 25))
>>> X_train.shape, X_test.shape
((6833, 25), (2278, 25))
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(Xm, Ym, random_state=9)
>>> X_train.shape, X_test.shape
((6833, 25), (2278, 25))
>>> y_train.shape, y_test.shape
((6833,), (2278,))
>>>
>>> Xp.shape
(10000, 42)
>>>
>>> X_train.shape, X_test.shape, Xp.shape
((6833, 25), (2278, 25), (10000, 42))
>>> y_train.shape, y_test.shape, Yp.shape
((6833,), (2278,), (10000,))
>>>
>>> scaler = StandardScaler().fit(X_train)
>>> X_train = scaler.transform(X_train)
>>> X_test = scaler.transform(X_test
...     [B
KeyboardInterrupt
>>> Xp = scaler.transform(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py", line 681, in transform
    estimator=self, dtype=FLOAT_DTYPES)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 433, in check_array
    array = np.array(array, dtype=dtype, order=order, copy=copy)
ValueError: could not convert string to float: 'F1+L'
>>>
>>> scaler = StandardScaler().fit(X_train)
>>> X_train = scaler.transform(X_train)
>>> X_test = scaler.transform(X_test)
>>> Xp = scaler.transform(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py", line 681, in transform
    estimator=self, dtype=FLOAT_DTYPES)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 433, in check_array
    array = np.array(array, dtype=dtype, order=order, copy=copy)
ValueError: could not convert string to float: 'F1+L'
>>> X_train.colnames()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'colnames'
>>> X_train
array([[-0.33487759, -0.05098011, -0.36242974, ..., -0.35886933,
        -0.47601042, -0.19687931],
       [ 0.33505073, -0.05098011, -0.36242974, ...,  0.2146426 ,
        -0.53710566, -0.70987361],
       [-0.54673489, -0.05098011, -0.36242974, ..., -0.71456471,
         0.94084272,  0.60739373],
       ...,
       [ 0.72401197, -0.05098011, -0.36242974, ...,  2.10327722,
         3.02732615, -1.05539085],
       [ 0.22558648, -0.05098011, -0.36242974, ..., -0.02159655,
        -0.65318916,  1.58954275],
       [-0.10035122, -0.05098011, -0.36242974, ..., -0.53827161,
        -0.17913488,  0.73316573]])
>>> X_train[1:20]
array([[ 0.33505073, -0.05098011, -0.36242974, -0.14926595, -0.03764888,
        -0.24293315,  0.57862884,  0.46694088, -0.60747773,  0.18636956,
        -0.06308962,  0.18628483, -0.22449578, -0.23319858, -0.37887029,
         0.13137941,  0.30021112, -0.0171109 ,  0.        , -0.03394986,
        -0.07005286, -0.01671905,  0.2146426 , -0.53710566, -0.70987361],
       [-0.54673489, -0.05098011, -0.36242974, -0.14926595, -0.04914833,
        -1.25667866, -0.16726516,  0.186447  ,  1.79278278,  1.38563974,
        -0.26102895,  0.18628483, -0.22449578, -0.23319858, -0.03324174,
        -0.10638103, -0.61230783, -0.0171109 ,  0.        , -0.37670897,
        -0.07005286,  0.3053514 , -0.71456471,  0.94084272,  0.60739373],
       [-0.7002583 , -0.05098011, -0.36242974, -0.14926595, -0.04914833,
        -1.34883734,  0.11494996,  0.37836386, -0.03261954, -0.18862882,
        -0.06308962, -0.4987982 , -0.22449578, -0.23319858, -0.53597417,
        -0.41207303, -0.61230783, -0.0171109 ,  0.        , -0.51381262,
        -0.07005286, -0.79295336, -0.71049663, -0.53651192,  0.86178374],
       [-0.46631997, -0.05098011, -0.36242974, -0.14926595, -0.0364768 ,
         0.86297105, -0.340936  , -0.06452121, -0.64808891, -0.60468982,
         2.70806101, -0.15625668, -0.22449578, -0.23319858, -0.6302365 ,
        -0.58190191, -0.61230783, -0.0171109 ,  0.        , -0.23960533,
        -0.07005286, -0.1471456 , -0.56947958, -0.81968916,  0.0540079 ],
       [-0.55186557, -0.05098011, -0.18732463, -0.14926595, -0.04914833,
        -0.5194092 ,  0.28379661, -0.5959833 ,  0.62878247,  0.38177935,
        -0.32288499, -0.4987982 , -0.22449578, -0.23319858, -0.5045534 ,
        -0.4460388 , -0.15604835, -0.0171109 ,  0.        , -0.51381262,
        -0.07005286, -0.79295336,  0.20519884, -0.2906127 , -0.16185143],
       [-0.62456242, -0.05098011,  4.30703987, -0.14926595, -0.04169869,
        -0.70372656, -0.13832002, -0.28596375, -0.54761126, -0.63276941,
        -0.17855423, -0.15625668, -0.22449578, -0.23319858, -0.34744951,
        -0.20827836, -0.15604835, -0.0171109 ,  0.        , -0.30815715,
        -0.07005286, -0.71509018, -0.64808068, -0.45879213, -2.5102521 ],
       [-0.00930742, -0.05098011, -0.36242974, -0.14926595, -0.03642069,
         0.586495  , -1.11087672, -1.34888792, -0.48879507, -0.58204499,
         0.39876882,  0.52882635, -0.22449578, -0.23319858, -0.37887029,
        -0.07241526, -0.61230783, -0.0171109 ,  0.        ,  0.24025744,
        -0.07005286,  0.36529007, -0.4124808 , -0.43752797,  0.87940714],
       [-0.2847028 , -0.05098011, -0.36242974, -0.14926595, -0.03908904,
        -1.44099603,  1.00139487,  0.49646655, -0.64808891, -0.71942362,
         0.1101073 , -0.4987982 , -0.22449578, -0.23319858, -0.59881573,
        -0.58190191,  0.30021112, -0.0171109 ,  0.        , -0.30815715,
        -0.07005286, -0.20854563, -0.1130702 , -0.72550597, -1.6026468 ],
       [-0.58113542, -0.05098011, -0.36242974, -0.14926595, -0.03127624,
         0.12570158, -0.64485997,  0.11263282, -0.01266405,  0.30351881,
         0.97609187, -0.4987982 , -0.22449578, -0.23319858, -0.59881573,
        -0.4460388 , -0.61230783, -0.0171109 ,  0.        , -0.23960533,
        -0.07005286,  0.27954923, -0.47977845, -0.66574239,  1.114696  ],
       [-0.51922917, -0.05098011, -0.36242974, -0.14926595, -0.04914833,
         0.95512973,  1.12802986,  1.13126849,  0.48776366,  0.26016151,
         0.97609187, -0.4987982 , -0.22449578, -0.23319858, -0.6302365 ,
        -0.58190191, -0.15604835, -0.0171109 ,  0.        , -0.51381262,
        -0.07005286, -0.75361076, -0.71456471, -0.61915123,  0.40373943],
       [ 0.06468313, -0.05098011, -0.36242974, -0.14926595, -0.04914833,
         0.49433632,  0.26691194,  0.37836386,  0.39484809,  0.18002901,
         4.44003015,  0.18628483, -0.22449578, -0.23319858, -0.59881573,
        -0.54793614, -0.61230783, -0.0171109 ,  0.        , -0.37670897,
        -0.07005286,  0.10630935, -0.16217773, -0.67634479,  0.33116603],
       [ 0.32565674, -0.05098011,  0.45472744, -0.14926595, -0.03219802,
        -0.5194092 ,  0.35808914, -0.10880972, -0.2857392 , -0.40692498,
        -0.12607031, -0.15625668,  1.39192117, -0.23319858, -0.44171184,
        -0.58190191,  0.30021112, -0.0171109 ,  0.        , -0.1710535 ,
        -0.07005286, -0.33488493,  0.88825833, -0.58817527, -0.38739816],
       [ 0.93730998, -0.05098011, -0.12895626, -0.14926595, -0.03338745,
        -0.15077446, -0.04714282,  0.99840297, -0.61377947, -0.30577807,
        -0.55793795, -0.15625668, -0.22449578, -0.23319858,  0.24954526,
         1.15035272, -0.15604835, -0.0171109 ,  0.        ,  0.65156838,
        -0.07005286,  0.4469665 ,  0.50004738, -0.12972772,  1.31347924],
       [-0.0226119 , -0.05098011,  0.10451722, -0.14926595, -0.04480416,
         1.41592315, -0.84747595, -0.5959833 , -0.55041203, -0.61676706,
         2.70806101, -0.4987982 , -0.22449578, -0.23319858, -0.6302365 ,
        -0.51397036, -0.61230783, -0.0171109 ,  0.        , -0.30815715,
        -0.07005286, -0.39610229, -0.29020598, -0.75321639, -0.50600039],
       [-0.59186721, -0.05098011, -0.36242974, -0.14926595, -0.0259842 ,
         1.23160578,  0.04563397, -0.17524248,  0.62591168,  0.37930351,
        -0.40060155,  0.52882635, -0.22449578, -0.23319858,  0.43806992,
        -0.14034681,  0.7564706 , -0.0171109 ,  0.        ,  0.03460197,
        -0.07005286,  0.41767076, -0.71456471,  2.56895494,  0.36553714],
       [ 1.97960202, -0.05098011, -0.36242974, -0.14926595, -0.02826529,
         1.1394471 , -0.67248942, -0.5959833 , -0.37396347, -0.09170895,
        -0.4409738 , -0.4987982 , -0.22449578, -0.23319858, -0.34744951,
         0.02948208, -0.61230783, -0.0171109 ,  0.        ,  0.65156838,
        -0.07005286,  0.4487247 ,  2.70323167, -0.43278658,  0.86796835],
       [-0.67195871, -0.05098011, -0.36242974, -0.14926595,  0.0090225 ,
         1.32376446, -1.45532389, -1.12744538, -0.6487891 ,  2.03720697,
         2.70806101, -0.15625668, -0.22449578, -0.23319858, -0.6302365 ,
        -0.27620992, -0.61230783, -0.0171109 ,  0.        ,  0.10315379,
        -0.07005286,  0.1859079 , -0.66080795, -0.84432913, -0.33233871],
       [ 1.50878516, -0.05098011,  0.60064837, -0.14926595, -0.02841991,
        -1.44099603,  0.5398807 ,  0.186447  ,  0.14873036, -0.03222853,
        -0.11778338,  1.55645089,  1.39192117,  1.3413527 ,  0.24954526,
         0.16534518,  2.12524902, -0.0171109 ,  0.        ,  1.61129391,
        -0.07005286,  0.34921509,  0.34287425,  0.45095271, -0.75836534],
       [-0.38727225, -0.05098011, -0.36242974, -0.14926595, -0.04914833,
         1.32376446,  0.22221724,  0.37836386,  1.10120247,  2.14167512,
        -0.35999861,  0.52882635, -0.22449578, -0.23319858,  0.40664914,
         0.70879762, -0.61230783, -0.0171109 ,  0.        , -0.30815715,
        -0.07005286, -0.79295336, -0.09511254,  2.0612105 ,  0.88882088]])
>>> Xp.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 89976 to 99975
Data columns (total 42 columns):
uuid                                   10000 non-null object
account_amount_added_12_24m            10000 non-null int64
account_days_in_dc_12_24m              8847 non-null float64
account_days_in_rem_12_24m             8847 non-null float64
account_days_in_term_12_24m            8847 non-null float64
account_incoming_debt_vs_paid_0_24m    4042 non-null float64
account_status                         4561 non-null float64
account_worst_status_0_3m              4561 non-null float64
account_worst_status_12_24m            3294 non-null float64
account_worst_status_3_6m              4236 non-null float64
account_worst_status_6_12m             3963 non-null float64
age                                    10000 non-null int64
avg_payment_span_0_12m                 7632 non-null float64
avg_payment_span_0_3m                  5077 non-null float64
merchant_category                      10000 non-null object
merchant_group                         10000 non-null object
has_paid                               10000 non-null bool
max_paid_inv_0_12m                     10000 non-null float64
max_paid_inv_0_24m                     10000 non-null float64
name_in_email                          10000 non-null object
num_active_div_by_paid_inv_0_12m       7719 non-null float64
num_active_inv                         10000 non-null int64
num_arch_dc_0_12m                      10000 non-null int64
num_arch_dc_12_24m                     10000 non-null int64
num_arch_ok_0_12m                      10000 non-null int64
num_arch_ok_12_24m                     10000 non-null int64
num_arch_rem_0_12m                     10000 non-null int64
num_arch_written_off_0_12m             8227 non-null float64
num_arch_written_off_12_24m            8227 non-null float64
num_unpaid_bills                       10000 non-null int64
status_last_archived_0_24m             10000 non-null int64
status_2nd_last_archived_0_24m         10000 non-null int64
status_3rd_last_archived_0_24m         10000 non-null int64
status_max_archived_0_6_months         10000 non-null int64
status_max_archived_0_12_months        10000 non-null int64
status_max_archived_0_24_months        10000 non-null int64
recovery_debt                          10000 non-null int64
sum_capital_paid_account_0_12m         10000 non-null int64
sum_capital_paid_account_12_24m        10000 non-null int64
sum_paid_inv_0_12m                     10000 non-null int64
time_hours                             10000 non-null float64
worst_status_active_inv                3025 non-null float64
dtypes: bool(1), float64(18), int64(19), object(4)
memory usage: 3.2+ MB
>>> Xp = Xp.fillna(0).as_matix()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/pandas/core/generic.py", line 3081, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'as_matix'
>>>
>>> Xp = Xp.fillna(0).as_matrix()
>>> Xp
array([['6f6e6c6a-2081-4e6b-8eb3-4fd89b54b2d7', 0, 0.0, ..., 27157,
        19.8955555555556, 0.0],
       ['f6f6d9f3-ef2b-4329-a388-c6a687f27e70', 0, 0.0, ..., 0,
        0.23666666666666697, 0.0],
       ['e9c39869-1bc5-4375-b627-a2df70b445ea', 50956, 0.0, ..., 93760,
        20.3327777777778, 0.0],
       ...,
       ['b22e21ea-b1b2-4df3-b236-0ff6d5fdc0d8', 45671, 0.0, ..., 3100,
        2.18527777777778, 0.0],
       ['bafcab15-9898-479c-b729-c9dda7edb78f', 56102, 0.0, ..., 34785,
        9.725277777777778, 0.0],
       ['ac88f18c-96a6-49bc-9e9d-a780225914af', 0, 0.0, ..., 30602,
        11.5852777777778, 0.0]], dtype=object)
>>>
>>>
>>>
>>>
>>>
>>>
>>>
>>>
>>>
>>>
>>> 0 == dfp.default.notnull().sum() #True
True
>>> Yp = dfp['default'] # Y for prediction
>>> Xp = dfp.drop('default', axis=1) # X for prediction
>>> # dfm contains all rows that have deafult=1 or 0
... dfm = raw_data[pd.notnull(raw_data.default)]
>>> 0 == dfm.default.isnull().sum() #True
True
>>> dfm.isnull().sum().sum() > 0 #true
True
>>> # qq handling NA choice 1: ignoring it. AND for  dfp set NA in X to 0
... dfm = dfm.dropna()
>>> Xp = Xp.fillna(0)
>>> # cehck fraction of defaults
... dfm['default'].mean() #concl: 0.014 so very few deafults
0.009987926682032707
>>>
>>> # Select Y for out model "Ym"
... Ym = dfm['default']
>>> numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
>>> Xm = dfm[numerical_variables].as_matrix()
>>> Xp = dfp[numerical_variables].as_matrix()
>>> # qq increase to also include categorical
... Ym.shape[0] == Xm.shape[0] #check number of rows are the same
True
>>> # inside dfm we split into X_train and X_test  (70% and 30%)
... # use random_state for reproducibility
... X_train, X_test, y_train, y_test = train_test_split(Xm, Ym, random_state=9)
>>> # check shapes look ok:
... X_train.shape, X_test.shape, Xp.shape
((6833, 25), (2278, 25), (10000, 25))
>>> y_train.shape, y_test.shape, Yp.shape
((6833,), (2278,), (10000,))
>>> data_raw.shape
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'data_raw' is not defined
>>> len(raw_data)
99976
>>> (X_train.shape[0] + X_test[0]) / len(data_raw)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'data_raw' is not defined
>>> (X_train.shape[0] + X_test[0])
array([ 539594.        ,    6833.        ,    6833.        ,
          6833.        ,    6833.25453969,    6873.        ,
          6856.09333333,    6859.63636364,   12128.        ,
         23633.        ,    6833.0125    ,    6834.        ,
          6833.        ,    6833.        ,    6907.        ,
          6905.        ,    6834.        ,    6833.        ,
          6833.        ,    6835.        ,    6833.        ,
        124934.        ,   86810.        ,  178608.        ,
          6840.72111111])
>>> X_train.shape[0]
6833
>>> (X_train.shape[0] + X_test.shape[0]) / len(data_raw)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'data_raw' is not defined
>>> (X_train.shape[0] + X_test.shape[0])
9111
>>> len(data_raw)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'data_raw' is not defined
>>> len(raw_data)
99976
>>> (X_train.shape[0] + X_test.shape[0]) / len(raw_data)
0.09113187164919581
>>>
>>> scaler = StandardScaler().fit(X_train)
>>> X_train = scaler.transform(X_train)
>>> X_test = scaler.transform(X_test)
>>> Xp = scaler.transform(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/data.py", line 681, in transform
    estimator=self, dtype=FLOAT_DTYPES)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 453, in check_array
    _assert_all_finite(array)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 44, in _assert_all_finite
    " or a value too large for %r." % X.dtype)
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
>>> Xp
array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   2.71570000e+04,   1.98955556e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   0.00000000e+00,   2.36666667e-01],
       [  5.09560000e+04,   0.00000000e+00,   7.70000000e+01, ...,
          3.98460000e+04,   9.37600000e+04,   2.03327778e+01],
       ...,
       [  4.56710000e+04,   0.00000000e+00,   2.00000000e+01, ...,
          1.96270000e+04,   3.10000000e+03,   2.18527778e+00],
       [  5.61020000e+04,   0.00000000e+00,   0.00000000e+00, ...,
          5.61800000e+04,   3.47850000e+04,   9.72527778e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   3.06020000e+04,   1.15852778e+01]])
>>> Xp.nansum()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'nansum'
>>> np.nansum(Xp)
881718072.68651581
>>> knn = KNeighborsClassifier(n_neighbors=5)
>>> knn.fit(X_train, Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
>>> Y_pred = knn.predict(X_test)
>>> print(metrics.accuracy_score(Y_test, Y_pred))
0.990342405619
>>> # checking classification accuracy of KNN with K=5
... knn = KNeighborsClassifier(n_neighbors=5)
>>> knn.fit(X_train, Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
>>> Y_hat = knn.predict(X_test)
>>> print(metrics.accuracy_score(Y_test, y_hat))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_hat' is not defined
>>> y_hat = knn.predict(X_test)
>>> print(metrics.accuracy_score(Y_test, y_hat))
0.990342405619
>>> print(metrics.accuracy_score(y_test, y_hat))
0.990342405619
>>> y_p = knn.predcit(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'KNeighborsClassifier' object has no attribute 'predcit'
>>> y_p = knn.predict(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/neighbors/classification.py", line 143, in predict
    X = check_array(X, accept_sparse='csr')
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 453, in check_array
    _assert_all_finite(array)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 44, in _assert_all_finite
    " or a value too large for %r." % X.dtype)
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
>>> Xp
array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   2.71570000e+04,   1.98955556e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   0.00000000e+00,   2.36666667e-01],
       [  5.09560000e+04,   0.00000000e+00,   7.70000000e+01, ...,
          3.98460000e+04,   9.37600000e+04,   2.03327778e+01],
       ...,
       [  4.56710000e+04,   0.00000000e+00,   2.00000000e+01, ...,
          1.96270000e+04,   3.10000000e+03,   2.18527778e+00],
       [  5.61020000e+04,   0.00000000e+00,   0.00000000e+00, ...,
          5.61800000e+04,   3.47850000e+04,   9.72527778e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   3.06020000e+04,   1.15852778e+01]])
>>> Xp.nansum()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'nansum'
>>> np.nansum(Xp)
881718072.68651581
>>> np.nansum(X_train)
-3.311839691377827e-11
>>> print(Xp)
[[  0.00000000e+00   0.00000000e+00   0.00000000e+00 ...,   0.00000000e+00
    2.71570000e+04   1.98955556e+01]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00 ...,   0.00000000e+00
    0.00000000e+00   2.36666667e-01]
 [  5.09560000e+04   0.00000000e+00   7.70000000e+01 ...,   3.98460000e+04
    9.37600000e+04   2.03327778e+01]
 ...,
 [  4.56710000e+04   0.00000000e+00   2.00000000e+01 ...,   1.96270000e+04
    3.10000000e+03   2.18527778e+00]
 [  5.61020000e+04   0.00000000e+00   0.00000000e+00 ...,   5.61800000e+04
    3.47850000e+04   9.72527778e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00 ...,   0.00000000e+00
    3.06020000e+04   1.15852778e+01]]
>>> y_p = knn.predict(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/neighbors/classification.py", line 143, in predict
    X = check_array(X, accept_sparse='csr')
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 453, in check_array
    _assert_all_finite(array)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 44, in _assert_all_finite
    " or a value too large for %r." % X.dtype)
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
>>> numerical_variables
2             account_amount_added_12_24m
3               account_days_in_dc_12_24m
4              account_days_in_rem_12_24m
5             account_days_in_term_12_24m
6     account_incoming_debt_vs_paid_0_24m
12                                    age
13                 avg_payment_span_0_12m
14                  avg_payment_span_0_3m
18                     max_paid_inv_0_12m
19                     max_paid_inv_0_24m
21       num_active_div_by_paid_inv_0_12m
22                         num_active_inv
23                      num_arch_dc_0_12m
24                     num_arch_dc_12_24m
25                      num_arch_ok_0_12m
26                     num_arch_ok_12_24m
27                     num_arch_rem_0_12m
28             num_arch_written_off_0_12m
29            num_arch_written_off_12_24m
30                       num_unpaid_bills
37                          recovery_debt
38         sum_capital_paid_account_0_12m
39        sum_capital_paid_account_12_24m
40                     sum_paid_inv_0_12m
41                             time_hours
Name: Variable, dtype: object
>>> Xm.shape
(9111, 25)
>>> numerical_variables.shape
(25,)
>>> y_p = knn.predict(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/neighbors/classification.py", line 143, in predict
    X = check_array(X, accept_sparse='csr')
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 453, in check_array
    _assert_all_finite(array)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 44, in _assert_all_finite
    " or a value too large for %r." % X.dtype)
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
>>>
>>> y_hat
array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
>>> y_hat.sum()
2.0
>>> y_hat.sum(), y_hat.count()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'count'
>>> y_hat.sum(), y_hat.count_values()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'count_values'
>>> y_hat.count_values()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'count_values'
>>> y_hat.count()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'count'
>>> y_hat.sum()
2.0
>>> leng(y_hat)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'leng' is not defined
>>> len(y_hat)
2278
>>> y_hat.sum(), len(y_hat)
(2.0, 2278)
>>> print(y_hat.sum(), len(y_hat))
2.0 2278
>>> X_train
array([[-0.33487759, -0.05098011, -0.36242974, ..., -0.35886933,
        -0.47601042, -0.19687931],
       [ 0.33505073, -0.05098011, -0.36242974, ...,  0.2146426 ,
        -0.53710566, -0.70987361],
       [-0.54673489, -0.05098011, -0.36242974, ..., -0.71456471,
         0.94084272,  0.60739373],
       ...,
       [ 0.72401197, -0.05098011, -0.36242974, ...,  2.10327722,
         3.02732615, -1.05539085],
       [ 0.22558648, -0.05098011, -0.36242974, ..., -0.02159655,
        -0.65318916,  1.58954275],
       [-0.10035122, -0.05098011, -0.36242974, ..., -0.53827161,
        -0.17913488,  0.73316573]])
>>> np.nansum(X_train)
-3.311839691377827e-11
>>> np.nansum(Xp)
881718072.68651581
>>>
>>> dfp.drop('default', axis=1).fillna.()
  File "<stdin>", line 1
    dfp.drop('default', axis=1).fillna.()
                                       ^
SyntaxError: invalid syntax
>>> dfp.drop('default', axis=1).fillna(0)
                                       uuid  account_amount_added_12_24m  \
89976  6f6e6c6a-2081-4e6b-8eb3-4fd89b54b2d7                            0
89977  f6f6d9f3-ef2b-4329-a388-c6a687f27e70                            0
89978  e9c39869-1bc5-4375-b627-a2df70b445ea                        50956
89979  6beb88a3-9641-4381-beb6-c9a208664dd0                        35054
89980  bb89b735-72fe-42a4-ba06-d63be0f4ca36                            0
89981  e4eede99-76a3-4437-a540-3059a1eff67c                            0
89982  a2af8d9e-9f81-4185-8fff-b2ec49d681a6                            0
89983  ec910486-1e66-402a-80f2-08c6f04a9a1b                       118135
89984  08973cf0-646a-4fa7-9f1f-d03f76ffd59c                            0
89985  0591fb4e-5b48-4bac-bce7-f2d5d141e976                         5708
89986  0d08fe69-ca57-433d-a6db-83970f6d11d8                            0
89987  53cfc224-c727-467c-b11b-585ce4137d83                        12571
89988  7d7c6bed-ce10-45c3-8ef4-55e84ae7f9d9                         8002
89989  d1087a43-7eb0-47ad-839c-14a0e8c5c0da                            0
89990  5c678d63-1da0-42e8-aed0-bafda88ffe98                        31175
89991  8d1bd7fd-d065-425f-a35f-044a7af0217d                            0
89992  a0e7a004-a491-491b-a95c-d17de6b8b740                        15162
89993  a111eeef-b3e1-4b2e-8d70-eacad33d87d4                            0
89994  484f9dcf-c62e-4903-a5e6-7a0aff8006d4                            0
89995  42df7daa-098c-49f6-b39b-96873d44fbc5                            0
89996  ee10e62a-a3e2-418d-adaf-6c81e96c87a2                            0
89997  00463b34-2e21-4966-9bbe-64a218e6f73d                        50059
89998  8f7cb7a6-f30b-453c-9c00-b97ccf33c4dd                            0
89999  65018d72-260a-4fe0-8149-11b4bb374114                            0
90000  4f1b8a2c-d178-4607-92ac-13457059047d                            0
90001  bea496e1-0b89-4c0b-b0f6-4b473dd5197d                            0
90002  068370bd-153c-4700-af73-41bba5a64992                            0
90003  09323f87-508f-4f84-99be-4510fad46dbb                         1295
90004  055cf2d1-8b9d-4cfb-b51c-bc62be31d3db                         3943
90005  163735a0-ff17-488e-a523-96b0635189b0                            0
...                                     ...                          ...
99946  ee3b2bd9-4e6b-457d-9de8-a6d79d8175b6                        18004
99947  c47e35d0-d572-45d0-ba33-d2f47dc7dc1a                            0
99948  f5fb4e0a-2dab-4453-85dc-9f3be38749c9                            0
99949  9e2acfd1-3e4b-4e33-995b-69c184c10871                            0
99950  0dd34e63-0a32-48f9-b83b-b5b2cd50370d                            0
99951  fcd79890-9006-4f20-a8fc-9281c307c860                            0
99952  4f4d907d-8c7e-4d61-82b3-eee25588f5ca                            0
99953  a85c288c-17c1-4024-bafd-9682729dd5ec                        78097
99954  b39629af-6dca-48f4-b4a3-780cc193ebcc                        28189
99955  95b381ea-4abc-43f6-9d2d-e2be654270e5                            0
99956  c7297e30-fdfc-4cba-9276-57e0762dde0b                            0
99957  b626a670-f761-4d06-9c2b-ec86390aee26                       133654
99958  5ed0b34c-f69f-4170-80a8-38a9922583f3                            0
99959  20077abe-8c0a-418b-b41f-f105529b46f8                            0
99960  9f00f01e-45ff-48d5-b99b-eb7ce76c3209                            0
99961  f059b965-9c71-4d97-a859-fa6af4a5f3d1                            0
99962  94765557-eda7-44e4-8183-b7cf5bbc9451                            0
99963  0cd0ce6e-989b-4f8c-b22c-af3624dfc749                            0
99964  a9d68ea6-d8b6-4210-8ac7-6461edea0408                            0
99965  07e03475-4945-4753-95d6-08a235645dc7                       112441
99966  34094999-c04b-4b8d-a404-84e612a24b11                        16642
99967  9610e28b-3ca1-45cf-8b83-2ae06e82dabc                        32355
99968  243fa082-bdf6-4a6f-81ee-41b28e6bb688                            0
99969  57e92a56-7145-4ea2-92d5-fb79388d82c1                            0
99970  0e5e16e1-7a8b-41a8-bc9d-8fac11e3abc1                        88405
99971  5c03bc63-ea65-4ffd-aa7b-95ea9a46db34                            0
99972  f8db22f4-9819-420c-abbc-9ddf1843176e                            0
99973  b22e21ea-b1b2-4df3-b236-0ff6d5fdc0d8                        45671
99974  bafcab15-9898-479c-b729-c9dda7edb78f                        56102
99975  ac88f18c-96a6-49bc-9e9d-a780225914af                            0

       account_days_in_dc_12_24m  account_days_in_rem_12_24m  \
89976                        0.0                         0.0
89977                        0.0                         0.0
89978                        0.0                        77.0
89979                        0.0                         0.0
89980                        0.0                         0.0
89981                        0.0                         0.0
89982                        0.0                         0.0
89983                        0.0                         0.0
89984                        0.0                         0.0
89985                        0.0                         0.0
89986                        0.0                         0.0
89987                        0.0                         0.0
89988                        0.0                         0.0
89989                        0.0                         0.0
89990                        0.0                        16.0
89991                        0.0                         0.0
89992                        0.0                         0.0
89993                        0.0                         0.0
89994                        0.0                         0.0
89995                        0.0                         0.0
89996                        0.0                         0.0
89997                        0.0                        34.0
89998                        0.0                         0.0
89999                        0.0                         0.0
90000                        0.0                         0.0
90001                        0.0                         0.0
90002                        0.0                         0.0
90003                        0.0                         4.0
90004                        0.0                         0.0
90005                        0.0                         0.0
...                          ...                         ...
99946                        0.0                         0.0
99947                        0.0                         0.0
99948                        0.0                         0.0
99949                        0.0                         0.0
99950                        0.0                         0.0
99951                        0.0                         0.0
99952                        0.0                         0.0
99953                        0.0                        51.0
99954                        0.0                        97.0
99955                        0.0                         0.0
99956                        0.0                         0.0
99957                        0.0                        48.0
99958                        0.0                         0.0
99959                        0.0                         0.0
99960                        0.0                         0.0
99961                        0.0                         0.0
99962                        0.0                         0.0
99963                        0.0                         0.0
99964                        0.0                         0.0
99965                        0.0                        17.0
99966                        0.0                        59.0
99967                        0.0                       125.0
99968                        0.0                         0.0
99969                      365.0                         0.0
99970                        0.0                        15.0
99971                        0.0                         0.0
99972                        0.0                         0.0
99973                        0.0                        20.0
99974                        0.0                         0.0
99975                        0.0                         0.0

       account_days_in_term_12_24m  account_incoming_debt_vs_paid_0_24m  \
89976                          0.0                             0.009135
89977                          0.0                             0.000000
89978                          0.0                             0.000000
89979                          0.0                             0.000000
89980                          0.0                             0.000000
89981                          0.0                             0.000000
89982                          0.0                             0.000000
89983                          0.0                             0.212976
89984                          0.0                             0.000000
89985                          0.0                             0.004301
89986                          0.0                             0.000000
89987                          0.0                             0.264635
89988                          0.0                             0.000000
89989                          0.0                             0.000011
89990                          0.0                             0.759593
89991                          0.0                             0.000000
89992                          0.0                             1.079245
89993                          0.0                             0.000000
89994                          0.0                             0.000000
89995                          0.0                             0.000000
89996                          0.0                             0.000000
89997                          0.0                             1.020338
89998                          0.0                             0.000000
89999                          0.0                             0.000000
90000                          0.0                             0.000000
90001                          0.0                             0.000000
90002                          0.0                             0.000000
90003                          0.0                             0.036302
90004                          0.0                             0.000000
90005                          0.0                             0.000000
...                            ...                                  ...
99946                          0.0                             0.000000
99947                          0.0                             0.000000
99948                          0.0                             0.000000
99949                          0.0                             0.000000
99950                          0.0                             0.000000
99951                          0.0                             0.000000
99952                          0.0                             0.000000
99953                          0.0                             0.907493
99954                          9.0                             0.000438
99955                          0.0                             0.992753
99956                          0.0                             0.000000
99957                          0.0                             0.147310
99958                          0.0                             0.000000
99959                          0.0                             0.000000
99960                          0.0                             0.000000
99961                          0.0                             0.000000
99962                          0.0                             0.974504
99963                          0.0                             0.000000
99964                          0.0                             0.000000
99965                          0.0                             0.577442
99966                          0.0                             0.000000
99967                         44.0                             0.665829
99968                          0.0                             0.000000
99969                          0.0                             0.371604
99970                          0.0                             0.672000
99971                          0.0                             0.000000
99972                          0.0                             0.004044
99973                          0.0                             0.705078
99974                          0.0                             0.064175
99975                          0.0                             0.000000

       account_status  account_worst_status_0_3m  account_worst_status_12_24m  \
89976             1.0                        1.0                          0.0
89977             0.0                        0.0                          0.0
89978             1.0                        1.0                          2.0
89979             1.0                        1.0                          1.0
89980             1.0                        2.0                          0.0
89981             0.0                        0.0                          0.0
89982             0.0                        0.0                          0.0
89983             1.0                        1.0                          1.0
89984             0.0                        0.0                          0.0
89985             1.0                        1.0                          1.0
89986             0.0                        0.0                          0.0
89987             1.0                        1.0                          1.0
89988             1.0                        1.0                          1.0
89989             1.0                        1.0                          0.0
89990             1.0                        1.0                          2.0
89991             0.0                        0.0                          0.0
89992             2.0                        2.0                          1.0
89993             1.0                        2.0                          0.0
89994             0.0                        0.0                          0.0
89995             0.0                        0.0                          0.0
89996             0.0                        0.0                          0.0
89997             1.0                        1.0                          2.0
89998             1.0                        1.0                          0.0
89999             0.0                        0.0                          0.0
90000             0.0                        0.0                          0.0
90001             0.0                        0.0                          0.0
90002             0.0                        0.0                          0.0
90003             1.0                        1.0                          2.0
90004             1.0                        1.0                          1.0
90005             0.0                        0.0                          0.0
...               ...                        ...                          ...
99946             1.0                        1.0                          1.0
99947             0.0                        0.0                          0.0
99948             0.0                        0.0                          0.0
99949             0.0                        0.0                          0.0
99950             0.0                        0.0                          0.0
99951             1.0                        1.0                          1.0
99952             1.0                        1.0                          0.0
99953             2.0                        2.0                          2.0
99954             1.0                        2.0                          3.0
99955             1.0                        1.0                          0.0
99956             0.0                        0.0                          0.0
99957             1.0                        1.0                          2.0
99958             0.0                        0.0                          0.0
99959             0.0                        0.0                          0.0
99960             0.0                        0.0                          0.0
99961             0.0                        0.0                          0.0
99962             1.0                        1.0                          0.0
99963             0.0                        0.0                          0.0
99964             0.0                        0.0                          0.0
99965             1.0                        1.0                          2.0
99966             1.0                        1.0                          2.0
99967             2.0                        2.0                          3.0
99968             0.0                        0.0                          0.0
99969             1.0                        1.0                          4.0
99970             2.0                        2.0                          2.0
99971             1.0                        1.0                          0.0
99972             1.0                        1.0                          0.0
99973             2.0                        2.0                          2.0
99974             1.0                        2.0                          1.0
99975             1.0                        1.0                          0.0

       account_worst_status_3_6m           ...            \
89976                        1.0           ...
89977                        0.0           ...
89978                        3.0           ...
89979                        1.0           ...
89980                        2.0           ...
89981                        0.0           ...
89982                        0.0           ...
89983                        1.0           ...
89984                        0.0           ...
89985                        1.0           ...
89986                        0.0           ...
89987                        1.0           ...
89988                        1.0           ...
89989                        1.0           ...
89990                        1.0           ...
89991                        0.0           ...
89992                        2.0           ...
89993                        1.0           ...
89994                        0.0           ...
89995                        0.0           ...
89996                        0.0           ...
89997                        1.0           ...
89998                        1.0           ...
89999                        0.0           ...
90000                        0.0           ...
90001                        0.0           ...
90002                        0.0           ...
90003                        1.0           ...
90004                        1.0           ...
90005                        0.0           ...
...                          ...           ...
99946                        1.0           ...
99947                        0.0           ...
99948                        0.0           ...
99949                        0.0           ...
99950                        0.0           ...
99951                        1.0           ...
99952                        0.0           ...
99953                        1.0           ...
99954                        1.0           ...
99955                        1.0           ...
99956                        0.0           ...
99957                        1.0           ...
99958                        0.0           ...
99959                        0.0           ...
99960                        0.0           ...
99961                        0.0           ...
99962                        1.0           ...
99963                        0.0           ...
99964                        0.0           ...
99965                        2.0           ...
99966                        1.0           ...
99967                        2.0           ...
99968                        0.0           ...
99969                        1.0           ...
99970                        1.0           ...
99971                        0.0           ...
99972                        1.0           ...
99973                        1.0           ...
99974                        2.0           ...
99975                        1.0           ...

       status_3rd_last_archived_0_24m  status_max_archived_0_6_months  \
89976                               1                               1
89977                               0                               0
89978                               2                               1
89979                               0                               2
89980                               0                               0
89981                               0                               0
89982                               1                               1
89983                               2                               2
89984                               1                               0
89985                               1                               1
89986                               1                               1
89987                               1                               1
89988                               0                               0
89989                               1                               1
89990                               1                               1
89991                               0                               1
89992                               0                               1
89993                               0                               0
89994                               1                               1
89995                               0                               0
89996                               1                               1
89997                               3                               0
89998                               1                               1
89999                               1                               0
90000                               1                               2
90001                               1                               1
90002                               2                               0
90003                               1                               2
90004                               1                               1
90005                               0                               0
...                               ...                             ...
99946                               1                               1
99947                               0                               0
99948                               1                               2
99949                               1                               1
99950                               0                               1
99951                               1                               2
99952                               2                               1
99953                               1                               2
99954                               1                               2
99955                               1                               1
99956                               1                               1
99957                               1                               1
99958                               0                               1
99959                               1                               1
99960                               1                               1
99961                               0                               0
99962                               1                               1
99963                               0                               0
99964                               0                               0
99965                               1                               1
99966                               2                               2
99967                               0                               1
99968                               1                               1
99969                               0                               0
99970                               0                               0
99971                               1                               1
99972                               1                               0
99973                               0                               0
99974                               1                               1
99975                               1                               2

       status_max_archived_0_12_months  status_max_archived_0_24_months  \
89976                                1                                1
89977                                0                                0
89978                                1                                3
89979                                2                                2
89980                                0                                0
89981                                0                                1
89982                                1                                1
89983                                2                                2
89984                                1                                1
89985                                2                                2
89986                                1                                1
89987                                2                                2
89988                                0                                1
89989                                1                                1
89990                                1                                1
89991                                1                                1
89992                                1                                1
89993                                1                                1
89994                                1                                1
89995                                0                                0
89996                                1                                1
89997                                1                                3
89998                                1                                1
89999                                1                                1
90000                                2                                2
90001                                1                                1
90002                                2                                2
90003                                2                                2
90004                                3                                3
90005                                0                                1
...                                ...                              ...
99946                                1                                1
99947                                1                                1
99948                                2                                2
99949                                2                                2
99950                                1                                1
99951                                2                                2
99952                                2                                2
99953                                2                                2
99954                                2                                2
99955                                1                                1
99956                                1                                1
99957                                2                                2
99958                                1                                1
99959                                1                                1
99960                                1                                1
99961                                0                                0
99962                                1                                1
99963                                0                                0
99964                                0                                0
99965                                1                                2
99966                                2                                2
99967                                1                                1
99968                                1                                1
99969                                0                                0
99970                                0                                1
99971                                1                                1
99972                                1                                1
99973                                0                                0
99974                                1                                1
99975                                2                                2

      recovery_debt sum_capital_paid_account_0_12m  \
89976             0                           8815
89977             0                              0
89978             0                          36163
89979             0                          62585
89980             0                          14295
89981             0                              0
89982             0                              0
89983             0                          37657
89984             0                              0
89985             0                           5707
89986             0                              0
89987             0                         121531
89988             0                              0
89989             0                          89579
89990             0                          30393
89991             0                              0
89992             0                              0
89993             0                           4645
89994             0                              0
89995             0                              0
89996             0                              0
89997             0                          25102
89998             0                              0
89999             0                              0
90000             0                              0
90001             0                              0
90002             0                              0
90003             0                              0
90004             0                              0
90005             0                              0
...             ...                            ...
99946             0                              0
99947             0                              0
99948             0                              0
99949             0                              0
99950             0                              0
99951             0                              0
99952             0                              0
99953             0                          45258
99954             0                          18585
99955             0                          21385
99956             0                              0
99957             0                         190459
99958             0                              0
99959             0                              0
99960             0                              0
99961             0                              0
99962             0                           6075
99963             0                              0
99964             0                              0
99965             0                          74306
99966             0                          12738
99967             0                          20633
99968             0                              0
99969             0                           7695
99970             0                          28870
99971             0                              0
99972             0                           7948
99973             0                          17447
99974             0                          18339
99975             0                              0

       sum_capital_paid_account_12_24m  sum_paid_inv_0_12m  time_hours  \
89976                                0               27157   19.895556
89977                                0                   0    0.236667
89978                            39846               93760   20.332778
89979                                0                1790    6.201111
89980                                0                   0    8.451111
89981                                0                6055   22.263056
89982                                0               64885   14.909444
89983                            42836               13483    4.866111
89984                                0                4790    9.260000
89985                                0              131364   12.881944
89986                                0               91572   22.881667
89987                            12480               59075   16.904444
89988                                0               20080   13.603611
89989                                0               20330   12.230833
89990                            21958               28224    9.644444
89991                                0                8286   22.792222
89992                                0               12650   10.028333
89993                                0                6155   16.773056
89994                                0               51025   20.314722
89995                                0                   0   12.371111
89996                                0                6400   15.847778
89997                                0                4880   13.968611
89998                                0               28520   15.454167
89999                                0               16095    9.953333
90000                                0                9906   21.117222
90001                                0               34485   10.009722
90002                                0               11805   22.726667
90003                            11997               39960   19.816389
90004                             5400               26020   19.588611
90005                                0                   0   16.371389
...                                ...                 ...         ...
99946                                0               15755   11.413889
99947                                0                 295   13.463889
99948                                0                4889   14.484167
99949                                0              136892   12.618056
99950                                0                1190   19.784722
99951                                0               39835    9.177500
99952                                0               12084   13.224167
99953                            18905               50092   21.173056
99954                             9669               25080   21.589444
99955                                0               22390   13.678333
99956                                0              128790    8.074167
99957                           145771              100650    7.998056
99958                                0               11850   18.946389
99959                                0               19165   22.544167
99960                                0               58008   15.111389
99961                                0                   0   15.560000
99962                                0               21110   18.788611
99963                                0                   0   10.114444
99964                                0                   0   12.478889
99965                            30838               97672   14.867778
99966                            10852               44244   17.471389
99967                            22355               10225   13.057500
99968                                0              276135   19.786944
99969                             1025                 895   11.290833
99970                            25771                   0    9.060833
99971                                0               60127   10.765556
99972                                0                4740   21.708333
99973                            19627                3100    2.185278
99974                            56180               34785    9.725278
99975                                0               30602   11.585278

      worst_status_active_inv
89976                     0.0
89977                     0.0
89978                     0.0
89979                     0.0
89980                     0.0
89981                     0.0
89982                     0.0
89983                     0.0
89984                     0.0
89985                     1.0
89986                     1.0
89987                     0.0
89988                     1.0
89989                     1.0
89990                     0.0
89991                     0.0
89992                     0.0
89993                     1.0
89994                     1.0
89995                     0.0
89996                     0.0
89997                     1.0
89998                     1.0
89999                     0.0
90000                     2.0
90001                     1.0
90002                     1.0
90003                     1.0
90004                     1.0
90005                     0.0
...                       ...
99946                     0.0
99947                     0.0
99948                     0.0
99949                     1.0
99950                     0.0
99951                     1.0
99952                     0.0
99953                     1.0
99954                     0.0
99955                     1.0
99956                     1.0
99957                     1.0
99958                     0.0
99959                     1.0
99960                     0.0
99961                     0.0
99962                     0.0
99963                     0.0
99964                     0.0
99965                     0.0
99966                     1.0
99967                     0.0
99968                     1.0
99969                     0.0
99970                     0.0
99971                     0.0
99972                     0.0
99973                     0.0
99974                     0.0
99975                     0.0

[10000 rows x 42 columns]
>>> dfp.drop('default', axis=1).fillna(0).head(2)
                                       uuid  account_amount_added_12_24m  \
89976  6f6e6c6a-2081-4e6b-8eb3-4fd89b54b2d7                            0
89977  f6f6d9f3-ef2b-4329-a388-c6a687f27e70                            0

       account_days_in_dc_12_24m  account_days_in_rem_12_24m  \
89976                        0.0                         0.0
89977                        0.0                         0.0

       account_days_in_term_12_24m  account_incoming_debt_vs_paid_0_24m  \
89976                          0.0                             0.009135
89977                          0.0                             0.000000

       account_status  account_worst_status_0_3m  account_worst_status_12_24m  \
89976             1.0                        1.0                          0.0
89977             0.0                        0.0                          0.0

       account_worst_status_3_6m           ...            \
89976                        1.0           ...
89977                        0.0           ...

       status_3rd_last_archived_0_24m  status_max_archived_0_6_months  \
89976                               1                               1
89977                               0                               0

       status_max_archived_0_12_months  status_max_archived_0_24_months  \
89976                                1                                1
89977                                0                                0

      recovery_debt sum_capital_paid_account_0_12m  \
89976             0                           8815
89977             0                              0

       sum_capital_paid_account_12_24m  sum_paid_inv_0_12m  time_hours  \
89976                                0               27157   19.895556
89977                                0                   0    0.236667

      worst_status_active_inv
89976                     0.0
89977                     0.0

[2 rows x 42 columns]
>>> dfp.drop('default', axis=1).fillna(0).tail(2)
                                       uuid  account_amount_added_12_24m  \
99974  bafcab15-9898-479c-b729-c9dda7edb78f                        56102
99975  ac88f18c-96a6-49bc-9e9d-a780225914af                            0

       account_days_in_dc_12_24m  account_days_in_rem_12_24m  \
99974                        0.0                         0.0
99975                        0.0                         0.0

       account_days_in_term_12_24m  account_incoming_debt_vs_paid_0_24m  \
99974                          0.0                             0.064175
99975                          0.0                             0.000000

       account_status  account_worst_status_0_3m  account_worst_status_12_24m  \
99974             1.0                        2.0                          1.0
99975             1.0                        1.0                          0.0

       account_worst_status_3_6m           ...            \
99974                        2.0           ...
99975                        1.0           ...

       status_3rd_last_archived_0_24m  status_max_archived_0_6_months  \
99974                               1                               1
99975                               1                               2

       status_max_archived_0_12_months  status_max_archived_0_24_months  \
99974                                1                                1
99975                                2                                2

      recovery_debt sum_capital_paid_account_0_12m  \
99974             0                          18339
99975             0                              0

       sum_capital_paid_account_12_24m  sum_paid_inv_0_12m  time_hours  \
99974                            56180               34785    9.725278
99975                                0               30602   11.585278

      worst_status_active_inv
99974                     0.0
99975                     0.0

[2 rows x 42 columns]
>>> dfp.drop('default', axis=1).fillna(0).describe()
       account_amount_added_12_24m  account_days_in_dc_12_24m  \
count                 10000.000000               10000.000000
mean                  12066.155400                   0.242600
std                   35643.592178                   6.783152
min                       0.000000                   0.000000
25%                       0.000000                   0.000000
50%                       0.000000                   0.000000
75%                    4454.500000                   0.000000
max                  963477.000000                 365.000000

       account_days_in_rem_12_24m  account_days_in_term_12_24m  \
count                10000.000000                 10000.000000
mean                     4.231300                     0.245900
std                     20.753286                     2.750744
min                      0.000000                     0.000000
25%                      0.000000                     0.000000
50%                      0.000000                     0.000000
75%                      0.000000                     0.000000
max                    365.000000                    61.000000

       account_incoming_debt_vs_paid_0_24m  account_status  \
count                         10000.000000     10000.00000
mean                              0.540876         0.47740
std                              14.049803         0.54048
min                               0.000000         0.00000
25%                               0.000000         0.00000
50%                               0.000000         0.00000
75%                               0.000513         1.00000
max                            1336.935484         2.00000

       account_worst_status_0_3m  account_worst_status_12_24m  \
count               10000.000000                 10000.000000
mean                    0.533400                     0.434900
std                     0.646472                     0.701293
min                     0.000000                     0.000000
25%                     0.000000                     0.000000
50%                     0.000000                     0.000000
75%                     1.000000                     1.000000
max                     4.000000                     4.000000

       account_worst_status_3_6m  account_worst_status_6_12m  \
count               10000.000000                10000.000000
mean                    0.501500                    0.493400
std                     0.653483                    0.683961
min                     0.000000                    0.000000
25%                     0.000000                    0.000000
50%                     0.000000                    0.000000
75%                     1.000000                    1.000000
max                     4.000000                    4.000000

                ...             status_3rd_last_archived_0_24m  \
count           ...                               10000.000000
mean            ...                                   0.742500
std             ...                                   0.641276
min             ...                                   0.000000
25%             ...                                   0.000000
50%             ...                                   1.000000
75%             ...                                   1.000000
max             ...                                   5.000000

       status_max_archived_0_6_months  status_max_archived_0_12_months  \
count                    10000.000000                     10000.000000
mean                         0.803500                         1.058400
std                          0.723422                         0.789083
min                          0.000000                         0.000000
25%                          0.000000                         1.000000
50%                          1.000000                         1.000000
75%                          1.000000                         2.000000
max                          3.000000                         5.000000

       status_max_archived_0_24_months  recovery_debt  \
count                     10000.000000   10000.000000
mean                          1.229600       3.954700
std                           0.832198     129.512655
min                           0.000000       0.000000
25%                           1.000000       0.000000
50%                           1.000000       0.000000
75%                           2.000000       0.000000
max                           5.000000   10230.000000

       sum_capital_paid_account_0_12m  sum_capital_paid_account_12_24m  \
count                    10000.000000                     10000.000000
mean                     10657.757600                      6210.353100
std                      26192.310929                     17616.399138
min                          0.000000                         0.000000
25%                          0.000000                         0.000000
50%                          0.000000                         0.000000
75%                       8643.750000                         0.000000
max                     490672.000000                    302385.000000

       sum_paid_inv_0_12m    time_hours  worst_status_active_inv
count        1.000000e+04  10000.000000             10000.000000
mean         3.916659e+04     15.270656                 0.337300
std          1.012026e+05      5.037400                 0.544204
min          0.000000e+00      0.008333                 0.000000
25%          2.570000e+03     11.553958                 0.000000
50%          1.593750e+04     15.714306                 0.000000
75%          4.301975e+04     19.500694                 1.000000
max          2.835652e+06     23.999722                 3.000000

[8 rows x 37 columns]
>>> dfp.drop('default', axis=1).fillna(0).inf8)
  File "<stdin>", line 1
    dfp.drop('default', axis=1).fillna(0).inf8)
                                              ^
SyntaxError: invalid syntax
>>> dfp.drop('default', axis=1).fillna(0).info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 89976 to 99975
Data columns (total 42 columns):
uuid                                   10000 non-null object
account_amount_added_12_24m            10000 non-null int64
account_days_in_dc_12_24m              10000 non-null float64
account_days_in_rem_12_24m             10000 non-null float64
account_days_in_term_12_24m            10000 non-null float64
account_incoming_debt_vs_paid_0_24m    10000 non-null float64
account_status                         10000 non-null float64
account_worst_status_0_3m              10000 non-null float64
account_worst_status_12_24m            10000 non-null float64
account_worst_status_3_6m              10000 non-null float64
account_worst_status_6_12m             10000 non-null float64
age                                    10000 non-null int64
avg_payment_span_0_12m                 10000 non-null float64
avg_payment_span_0_3m                  10000 non-null float64
merchant_category                      10000 non-null object
merchant_group                         10000 non-null object
has_paid                               10000 non-null bool
max_paid_inv_0_12m                     10000 non-null float64
max_paid_inv_0_24m                     10000 non-null float64
name_in_email                          10000 non-null object
num_active_div_by_paid_inv_0_12m       10000 non-null float64
num_active_inv                         10000 non-null int64
num_arch_dc_0_12m                      10000 non-null int64
num_arch_dc_12_24m                     10000 non-null int64
num_arch_ok_0_12m                      10000 non-null int64
num_arch_ok_12_24m                     10000 non-null int64
num_arch_rem_0_12m                     10000 non-null int64
num_arch_written_off_0_12m             10000 non-null float64
num_arch_written_off_12_24m            10000 non-null float64
num_unpaid_bills                       10000 non-null int64
status_last_archived_0_24m             10000 non-null int64
status_2nd_last_archived_0_24m         10000 non-null int64
status_3rd_last_archived_0_24m         10000 non-null int64
status_max_archived_0_6_months         10000 non-null int64
status_max_archived_0_12_months        10000 non-null int64
status_max_archived_0_24_months        10000 non-null int64
recovery_debt                          10000 non-null int64
sum_capital_paid_account_0_12m         10000 non-null int64
sum_capital_paid_account_12_24m        10000 non-null int64
sum_paid_inv_0_12m                     10000 non-null int64
time_hours                             10000 non-null float64
worst_status_active_inv                10000 non-null float64
dtypes: bool(1), float64(18), int64(19), object(4)
memory usage: 3.2+ MB
>>> dfp.drop('default', axis=1).fillna(0).info().isnull()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 89976 to 99975
Data columns (total 42 columns):
uuid                                   10000 non-null object
account_amount_added_12_24m            10000 non-null int64
account_days_in_dc_12_24m              10000 non-null float64
account_days_in_rem_12_24m             10000 non-null float64
account_days_in_term_12_24m            10000 non-null float64
account_incoming_debt_vs_paid_0_24m    10000 non-null float64
account_status                         10000 non-null float64
account_worst_status_0_3m              10000 non-null float64
account_worst_status_12_24m            10000 non-null float64
account_worst_status_3_6m              10000 non-null float64
account_worst_status_6_12m             10000 non-null float64
age                                    10000 non-null int64
avg_payment_span_0_12m                 10000 non-null float64
avg_payment_span_0_3m                  10000 non-null float64
merchant_category                      10000 non-null object
merchant_group                         10000 non-null object
has_paid                               10000 non-null bool
max_paid_inv_0_12m                     10000 non-null float64
max_paid_inv_0_24m                     10000 non-null float64
name_in_email                          10000 non-null object
num_active_div_by_paid_inv_0_12m       10000 non-null float64
num_active_inv                         10000 non-null int64
num_arch_dc_0_12m                      10000 non-null int64
num_arch_dc_12_24m                     10000 non-null int64
num_arch_ok_0_12m                      10000 non-null int64
num_arch_ok_12_24m                     10000 non-null int64
num_arch_rem_0_12m                     10000 non-null int64
num_arch_written_off_0_12m             10000 non-null float64
num_arch_written_off_12_24m            10000 non-null float64
num_unpaid_bills                       10000 non-null int64
status_last_archived_0_24m             10000 non-null int64
status_2nd_last_archived_0_24m         10000 non-null int64
status_3rd_last_archived_0_24m         10000 non-null int64
status_max_archived_0_6_months         10000 non-null int64
status_max_archived_0_12_months        10000 non-null int64
status_max_archived_0_24_months        10000 non-null int64
recovery_debt                          10000 non-null int64
sum_capital_paid_account_0_12m         10000 non-null int64
sum_capital_paid_account_12_24m        10000 non-null int64
sum_paid_inv_0_12m                     10000 non-null int64
time_hours                             10000 non-null float64
worst_status_active_inv                10000 non-null float64
dtypes: bool(1), float64(18), int64(19), object(4)
memory usage: 3.2+ MB
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'NoneType' object has no attribute 'isnull'
>>> dfp.drop('default', axis=1).fillna(0).isnull()
        uuid  account_amount_added_12_24m  account_days_in_dc_12_24m  \
89976  False                        False                      False
89977  False                        False                      False
89978  False                        False                      False
89979  False                        False                      False
89980  False                        False                      False
89981  False                        False                      False
89982  False                        False                      False
89983  False                        False                      False
89984  False                        False                      False
89985  False                        False                      False
89986  False                        False                      False
89987  False                        False                      False
89988  False                        False                      False
89989  False                        False                      False
89990  False                        False                      False
89991  False                        False                      False
89992  False                        False                      False
89993  False                        False                      False
89994  False                        False                      False
89995  False                        False                      False
89996  False                        False                      False
89997  False                        False                      False
89998  False                        False                      False
89999  False                        False                      False
90000  False                        False                      False
90001  False                        False                      False
90002  False                        False                      False
90003  False                        False                      False
90004  False                        False                      False
90005  False                        False                      False
...      ...                          ...                        ...
99946  False                        False                      False
99947  False                        False                      False
99948  False                        False                      False
99949  False                        False                      False
99950  False                        False                      False
99951  False                        False                      False
99952  False                        False                      False
99953  False                        False                      False
99954  False                        False                      False
99955  False                        False                      False
99956  False                        False                      False
99957  False                        False                      False
99958  False                        False                      False
99959  False                        False                      False
99960  False                        False                      False
99961  False                        False                      False
99962  False                        False                      False
99963  False                        False                      False
99964  False                        False                      False
99965  False                        False                      False
99966  False                        False                      False
99967  False                        False                      False
99968  False                        False                      False
99969  False                        False                      False
99970  False                        False                      False
99971  False                        False                      False
99972  False                        False                      False
99973  False                        False                      False
99974  False                        False                      False
99975  False                        False                      False

       account_days_in_rem_12_24m  account_days_in_term_12_24m  \
89976                       False                        False
89977                       False                        False
89978                       False                        False
89979                       False                        False
89980                       False                        False
89981                       False                        False
89982                       False                        False
89983                       False                        False
89984                       False                        False
89985                       False                        False
89986                       False                        False
89987                       False                        False
89988                       False                        False
89989                       False                        False
89990                       False                        False
89991                       False                        False
89992                       False                        False
89993                       False                        False
89994                       False                        False
89995                       False                        False
89996                       False                        False
89997                       False                        False
89998                       False                        False
89999                       False                        False
90000                       False                        False
90001                       False                        False
90002                       False                        False
90003                       False                        False
90004                       False                        False
90005                       False                        False
...                           ...                          ...
99946                       False                        False
99947                       False                        False
99948                       False                        False
99949                       False                        False
99950                       False                        False
99951                       False                        False
99952                       False                        False
99953                       False                        False
99954                       False                        False
99955                       False                        False
99956                       False                        False
99957                       False                        False
99958                       False                        False
99959                       False                        False
99960                       False                        False
99961                       False                        False
99962                       False                        False
99963                       False                        False
99964                       False                        False
99965                       False                        False
99966                       False                        False
99967                       False                        False
99968                       False                        False
99969                       False                        False
99970                       False                        False
99971                       False                        False
99972                       False                        False
99973                       False                        False
99974                       False                        False
99975                       False                        False

       account_incoming_debt_vs_paid_0_24m  account_status  \
89976                                False           False
89977                                False           False
89978                                False           False
89979                                False           False
89980                                False           False
89981                                False           False
89982                                False           False
89983                                False           False
89984                                False           False
89985                                False           False
89986                                False           False
89987                                False           False
89988                                False           False
89989                                False           False
89990                                False           False
89991                                False           False
89992                                False           False
89993                                False           False
89994                                False           False
89995                                False           False
89996                                False           False
89997                                False           False
89998                                False           False
89999                                False           False
90000                                False           False
90001                                False           False
90002                                False           False
90003                                False           False
90004                                False           False
90005                                False           False
...                                    ...             ...
99946                                False           False
99947                                False           False
99948                                False           False
99949                                False           False
99950                                False           False
99951                                False           False
99952                                False           False
99953                                False           False
99954                                False           False
99955                                False           False
99956                                False           False
99957                                False           False
99958                                False           False
99959                                False           False
99960                                False           False
99961                                False           False
99962                                False           False
99963                                False           False
99964                                False           False
99965                                False           False
99966                                False           False
99967                                False           False
99968                                False           False
99969                                False           False
99970                                False           False
99971                                False           False
99972                                False           False
99973                                False           False
99974                                False           False
99975                                False           False

       account_worst_status_0_3m  account_worst_status_12_24m  \
89976                      False                        False
89977                      False                        False
89978                      False                        False
89979                      False                        False
89980                      False                        False
89981                      False                        False
89982                      False                        False
89983                      False                        False
89984                      False                        False
89985                      False                        False
89986                      False                        False
89987                      False                        False
89988                      False                        False
89989                      False                        False
89990                      False                        False
89991                      False                        False
89992                      False                        False
89993                      False                        False
89994                      False                        False
89995                      False                        False
89996                      False                        False
89997                      False                        False
89998                      False                        False
89999                      False                        False
90000                      False                        False
90001                      False                        False
90002                      False                        False
90003                      False                        False
90004                      False                        False
90005                      False                        False
...                          ...                          ...
99946                      False                        False
99947                      False                        False
99948                      False                        False
99949                      False                        False
99950                      False                        False
99951                      False                        False
99952                      False                        False
99953                      False                        False
99954                      False                        False
99955                      False                        False
99956                      False                        False
99957                      False                        False
99958                      False                        False
99959                      False                        False
99960                      False                        False
99961                      False                        False
99962                      False                        False
99963                      False                        False
99964                      False                        False
99965                      False                        False
99966                      False                        False
99967                      False                        False
99968                      False                        False
99969                      False                        False
99970                      False                        False
99971                      False                        False
99972                      False                        False
99973                      False                        False
99974                      False                        False
99975                      False                        False

       account_worst_status_3_6m           ...             \
89976                      False           ...
89977                      False           ...
89978                      False           ...
89979                      False           ...
89980                      False           ...
89981                      False           ...
89982                      False           ...
89983                      False           ...
89984                      False           ...
89985                      False           ...
89986                      False           ...
89987                      False           ...
89988                      False           ...
89989                      False           ...
89990                      False           ...
89991                      False           ...
89992                      False           ...
89993                      False           ...
89994                      False           ...
89995                      False           ...
89996                      False           ...
89997                      False           ...
89998                      False           ...
89999                      False           ...
90000                      False           ...
90001                      False           ...
90002                      False           ...
90003                      False           ...
90004                      False           ...
90005                      False           ...
...                          ...           ...
99946                      False           ...
99947                      False           ...
99948                      False           ...
99949                      False           ...
99950                      False           ...
99951                      False           ...
99952                      False           ...
99953                      False           ...
99954                      False           ...
99955                      False           ...
99956                      False           ...
99957                      False           ...
99958                      False           ...
99959                      False           ...
99960                      False           ...
99961                      False           ...
99962                      False           ...
99963                      False           ...
99964                      False           ...
99965                      False           ...
99966                      False           ...
99967                      False           ...
99968                      False           ...
99969                      False           ...
99970                      False           ...
99971                      False           ...
99972                      False           ...
99973                      False           ...
99974                      False           ...
99975                      False           ...

       status_3rd_last_archived_0_24m  status_max_archived_0_6_months  \
89976                           False                           False
89977                           False                           False
89978                           False                           False
89979                           False                           False
89980                           False                           False
89981                           False                           False
89982                           False                           False
89983                           False                           False
89984                           False                           False
89985                           False                           False
89986                           False                           False
89987                           False                           False
89988                           False                           False
89989                           False                           False
89990                           False                           False
89991                           False                           False
89992                           False                           False
89993                           False                           False
89994                           False                           False
89995                           False                           False
89996                           False                           False
89997                           False                           False
89998                           False                           False
89999                           False                           False
90000                           False                           False
90001                           False                           False
90002                           False                           False
90003                           False                           False
90004                           False                           False
90005                           False                           False
...                               ...                             ...
99946                           False                           False
99947                           False                           False
99948                           False                           False
99949                           False                           False
99950                           False                           False
99951                           False                           False
99952                           False                           False
99953                           False                           False
99954                           False                           False
99955                           False                           False
99956                           False                           False
99957                           False                           False
99958                           False                           False
99959                           False                           False
99960                           False                           False
99961                           False                           False
99962                           False                           False
99963                           False                           False
99964                           False                           False
99965                           False                           False
99966                           False                           False
99967                           False                           False
99968                           False                           False
99969                           False                           False
99970                           False                           False
99971                           False                           False
99972                           False                           False
99973                           False                           False
99974                           False                           False
99975                           False                           False

       status_max_archived_0_12_months  status_max_archived_0_24_months  \
89976                            False                            False
89977                            False                            False
89978                            False                            False
89979                            False                            False
89980                            False                            False
89981                            False                            False
89982                            False                            False
89983                            False                            False
89984                            False                            False
89985                            False                            False
89986                            False                            False
89987                            False                            False
89988                            False                            False
89989                            False                            False
89990                            False                            False
89991                            False                            False
89992                            False                            False
89993                            False                            False
89994                            False                            False
89995                            False                            False
89996                            False                            False
89997                            False                            False
89998                            False                            False
89999                            False                            False
90000                            False                            False
90001                            False                            False
90002                            False                            False
90003                            False                            False
90004                            False                            False
90005                            False                            False
...                                ...                              ...
99946                            False                            False
99947                            False                            False
99948                            False                            False
99949                            False                            False
99950                            False                            False
99951                            False                            False
99952                            False                            False
99953                            False                            False
99954                            False                            False
99955                            False                            False
99956                            False                            False
99957                            False                            False
99958                            False                            False
99959                            False                            False
99960                            False                            False
99961                            False                            False
99962                            False                            False
99963                            False                            False
99964                            False                            False
99965                            False                            False
99966                            False                            False
99967                            False                            False
99968                            False                            False
99969                            False                            False
99970                            False                            False
99971                            False                            False
99972                            False                            False
99973                            False                            False
99974                            False                            False
99975                            False                            False

       recovery_debt  sum_capital_paid_account_0_12m  \
89976          False                           False
89977          False                           False
89978          False                           False
89979          False                           False
89980          False                           False
89981          False                           False
89982          False                           False
89983          False                           False
89984          False                           False
89985          False                           False
89986          False                           False
89987          False                           False
89988          False                           False
89989          False                           False
89990          False                           False
89991          False                           False
89992          False                           False
89993          False                           False
89994          False                           False
89995          False                           False
89996          False                           False
89997          False                           False
89998          False                           False
89999          False                           False
90000          False                           False
90001          False                           False
90002          False                           False
90003          False                           False
90004          False                           False
90005          False                           False
...              ...                             ...
99946          False                           False
99947          False                           False
99948          False                           False
99949          False                           False
99950          False                           False
99951          False                           False
99952          False                           False
99953          False                           False
99954          False                           False
99955          False                           False
99956          False                           False
99957          False                           False
99958          False                           False
99959          False                           False
99960          False                           False
99961          False                           False
99962          False                           False
99963          False                           False
99964          False                           False
99965          False                           False
99966          False                           False
99967          False                           False
99968          False                           False
99969          False                           False
99970          False                           False
99971          False                           False
99972          False                           False
99973          False                           False
99974          False                           False
99975          False                           False

       sum_capital_paid_account_12_24m  sum_paid_inv_0_12m  time_hours  \
89976                            False               False       False
89977                            False               False       False
89978                            False               False       False
89979                            False               False       False
89980                            False               False       False
89981                            False               False       False
89982                            False               False       False
89983                            False               False       False
89984                            False               False       False
89985                            False               False       False
89986                            False               False       False
89987                            False               False       False
89988                            False               False       False
89989                            False               False       False
89990                            False               False       False
89991                            False               False       False
89992                            False               False       False
89993                            False               False       False
89994                            False               False       False
89995                            False               False       False
89996                            False               False       False
89997                            False               False       False
89998                            False               False       False
89999                            False               False       False
90000                            False               False       False
90001                            False               False       False
90002                            False               False       False
90003                            False               False       False
90004                            False               False       False
90005                            False               False       False
...                                ...                 ...         ...
99946                            False               False       False
99947                            False               False       False
99948                            False               False       False
99949                            False               False       False
99950                            False               False       False
99951                            False               False       False
99952                            False               False       False
99953                            False               False       False
99954                            False               False       False
99955                            False               False       False
99956                            False               False       False
99957                            False               False       False
99958                            False               False       False
99959                            False               False       False
99960                            False               False       False
99961                            False               False       False
99962                            False               False       False
99963                            False               False       False
99964                            False               False       False
99965                            False               False       False
99966                            False               False       False
99967                            False               False       False
99968                            False               False       False
99969                            False               False       False
99970                            False               False       False
99971                            False               False       False
99972                            False               False       False
99973                            False               False       False
99974                            False               False       False
99975                            False               False       False

       worst_status_active_inv
89976                    False
89977                    False
89978                    False
89979                    False
89980                    False
89981                    False
89982                    False
89983                    False
89984                    False
89985                    False
89986                    False
89987                    False
89988                    False
89989                    False
89990                    False
89991                    False
89992                    False
89993                    False
89994                    False
89995                    False
89996                    False
89997                    False
89998                    False
89999                    False
90000                    False
90001                    False
90002                    False
90003                    False
90004                    False
90005                    False
...                        ...
99946                    False
99947                    False
99948                    False
99949                    False
99950                    False
99951                    False
99952                    False
99953                    False
99954                    False
99955                    False
99956                    False
99957                    False
99958                    False
99959                    False
99960                    False
99961                    False
99962                    False
99963                    False
99964                    False
99965                    False
99966                    False
99967                    False
99968                    False
99969                    False
99970                    False
99971                    False
99972                    False
99973                    False
99974                    False
99975                    False

[10000 rows x 42 columns]
>>> dfp.drop('default', axis=1).fillna(0).isnull().sum()
uuid                                   0
account_amount_added_12_24m            0
account_days_in_dc_12_24m              0
account_days_in_rem_12_24m             0
account_days_in_term_12_24m            0
account_incoming_debt_vs_paid_0_24m    0
account_status                         0
account_worst_status_0_3m              0
account_worst_status_12_24m            0
account_worst_status_3_6m              0
account_worst_status_6_12m             0
age                                    0
avg_payment_span_0_12m                 0
avg_payment_span_0_3m                  0
merchant_category                      0
merchant_group                         0
has_paid                               0
max_paid_inv_0_12m                     0
max_paid_inv_0_24m                     0
name_in_email                          0
num_active_div_by_paid_inv_0_12m       0
num_active_inv                         0
num_arch_dc_0_12m                      0
num_arch_dc_12_24m                     0
num_arch_ok_0_12m                      0
num_arch_ok_12_24m                     0
num_arch_rem_0_12m                     0
num_arch_written_off_0_12m             0
num_arch_written_off_12_24m            0
num_unpaid_bills                       0
status_last_archived_0_24m             0
status_2nd_last_archived_0_24m         0
status_3rd_last_archived_0_24m         0
status_max_archived_0_6_months         0
status_max_archived_0_12_months        0
status_max_archived_0_24_months        0
recovery_debt                          0
sum_capital_paid_account_0_12m         0
sum_capital_paid_account_12_24m        0
sum_paid_inv_0_12m                     0
time_hours                             0
worst_status_active_inv                0
dtype: int64
>>> dfp.drop('default', axis=1).fillna(0).isnull().sum().sum()
0
>>> Xpp = dfp.drop('default', axis=1).fillna(0).as_matrix()
>>> Xpp
array([['6f6e6c6a-2081-4e6b-8eb3-4fd89b54b2d7', 0, 0.0, ..., 27157,
        19.8955555555556, 0.0],
       ['f6f6d9f3-ef2b-4329-a388-c6a687f27e70', 0, 0.0, ..., 0,
        0.23666666666666697, 0.0],
       ['e9c39869-1bc5-4375-b627-a2df70b445ea', 50956, 0.0, ..., 93760,
        20.3327777777778, 0.0],
       ...,
       ['b22e21ea-b1b2-4df3-b236-0ff6d5fdc0d8', 45671, 0.0, ..., 3100,
        2.18527777777778, 0.0],
       ['bafcab15-9898-479c-b729-c9dda7edb78f', 56102, 0.0, ..., 34785,
        9.725277777777778, 0.0],
       ['ac88f18c-96a6-49bc-9e9d-a780225914af', 0, 0.0, ..., 30602,
        11.5852777777778, 0.0]], dtype=object)
>>> np.nansum(Xpp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/numpy/lib/nanfunctions.py", line 542, in nansum
    return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py", line 1834, in sum
    out=out, **kwargs)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/numpy/core/_methods.py", line 32, in _sum
    return umr_sum(a, axis, dtype, out, keepdims)
TypeError: must be str, not float
>>> Xpp = dfp[numerical_variables].fillna(0).isnull().sum().sum()
>>> (Xpp == Xp).sum()
131164
>>> (Xpp != Xp).sum()
118836
>>> dfp.head()
                                       uuid  default  \
89976  6f6e6c6a-2081-4e6b-8eb3-4fd89b54b2d7      NaN
89977  f6f6d9f3-ef2b-4329-a388-c6a687f27e70      NaN
89978  e9c39869-1bc5-4375-b627-a2df70b445ea      NaN
89979  6beb88a3-9641-4381-beb6-c9a208664dd0      NaN
89980  bb89b735-72fe-42a4-ba06-d63be0f4ca36      NaN

       account_amount_added_12_24m  account_days_in_dc_12_24m  \
89976                            0                        0.0
89977                            0                        0.0
89978                        50956                        0.0
89979                        35054                        0.0
89980                            0                        0.0

       account_days_in_rem_12_24m  account_days_in_term_12_24m  \
89976                         0.0                          0.0
89977                         0.0                          0.0
89978                        77.0                          0.0
89979                         0.0                          0.0
89980                         0.0                          0.0

       account_incoming_debt_vs_paid_0_24m  account_status  \
89976                             0.009135             1.0
89977                                  NaN             NaN
89978                             0.000000             1.0
89979                             0.000000             1.0
89980                             0.000000             1.0

       account_worst_status_0_3m  account_worst_status_12_24m  \
89976                        1.0                          NaN
89977                        NaN                          NaN
89978                        1.0                          2.0
89979                        1.0                          1.0
89980                        2.0                          NaN

                ...             status_3rd_last_archived_0_24m  \
89976           ...                                          1
89977           ...                                          0
89978           ...                                          2
89979           ...                                          0
89980           ...                                          0

       status_max_archived_0_6_months  status_max_archived_0_12_months  \
89976                               1                                1
89977                               0                                0
89978                               1                                1
89979                               2                                2
89980                               0                                0

       status_max_archived_0_24_months  recovery_debt  \
89976                                1              0
89977                                0              0
89978                                3              0
89979                                2              0
89980                                0              0

      sum_capital_paid_account_0_12m sum_capital_paid_account_12_24m  \
89976                           8815                               0
89977                              0                               0
89978                          36163                           39846
89979                          62585                               0
89980                          14295                               0

       sum_paid_inv_0_12m  time_hours  worst_status_active_inv
89976               27157   19.895556                      NaN
89977                   0    0.236667                      NaN
89978               93760   20.332778                      NaN
89979                1790    6.201111                      NaN
89980                   0    8.451111                      NaN

[5 rows x 43 columns]
>>> dfp.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 89976 to 99975
Data columns (total 43 columns):
uuid                                   10000 non-null object
default                                0 non-null float64
account_amount_added_12_24m            10000 non-null int64
account_days_in_dc_12_24m              8847 non-null float64
account_days_in_rem_12_24m             8847 non-null float64
account_days_in_term_12_24m            8847 non-null float64
account_incoming_debt_vs_paid_0_24m    4042 non-null float64
account_status                         4561 non-null float64
account_worst_status_0_3m              4561 non-null float64
account_worst_status_12_24m            3294 non-null float64
account_worst_status_3_6m              4236 non-null float64
account_worst_status_6_12m             3963 non-null float64
age                                    10000 non-null int64
avg_payment_span_0_12m                 7632 non-null float64
avg_payment_span_0_3m                  5077 non-null float64
merchant_category                      10000 non-null object
merchant_group                         10000 non-null object
has_paid                               10000 non-null bool
max_paid_inv_0_12m                     10000 non-null float64
max_paid_inv_0_24m                     10000 non-null float64
name_in_email                          10000 non-null object
num_active_div_by_paid_inv_0_12m       7719 non-null float64
num_active_inv                         10000 non-null int64
num_arch_dc_0_12m                      10000 non-null int64
num_arch_dc_12_24m                     10000 non-null int64
num_arch_ok_0_12m                      10000 non-null int64
num_arch_ok_12_24m                     10000 non-null int64
num_arch_rem_0_12m                     10000 non-null int64
num_arch_written_off_0_12m             8227 non-null float64
num_arch_written_off_12_24m            8227 non-null float64
num_unpaid_bills                       10000 non-null int64
status_last_archived_0_24m             10000 non-null int64
status_2nd_last_archived_0_24m         10000 non-null int64
status_3rd_last_archived_0_24m         10000 non-null int64
status_max_archived_0_6_months         10000 non-null int64
status_max_archived_0_12_months        10000 non-null int64
status_max_archived_0_24_months        10000 non-null int64
recovery_debt                          10000 non-null int64
sum_capital_paid_account_0_12m         10000 non-null int64
sum_capital_paid_account_12_24m        10000 non-null int64
sum_paid_inv_0_12m                     10000 non-null int64
time_hours                             10000 non-null float64
worst_status_active_inv                3025 non-null float64
dtypes: bool(1), float64(19), int64(19), object(4)
memory usage: 3.3+ MB
>>> yp = knn.predict(Xpp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/neighbors/classification.py", line 145, in predict
    neigh_dist, neigh_ind = self.kneighbors(X)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/neighbors/base.py", line 385, in kneighbors
    for s in gen_even_slices(X.shape[0], n_jobs)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 779, in __call__
    while self.dispatch_one_batch(iterator):
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 625, in dispatch_one_batch
    self._dispatch(tasks)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 588, in _dispatch
    job = self._backend.apply_async(batch, callback=cb)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py", line 111, in apply_async
    result = ImmediateResult(func)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py", line 332, in __init__
    self.results = batch()
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 131, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py", line 131, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
  File "sklearn/neighbors/binary_tree.pxi", line 1294, in sklearn.neighbors.kd_tree.BinaryTree.query
ValueError: query data dimension must match training data dimension
>>> Xpp.shape
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'int' object has no attribute 'shape'
>>> Xpp.shape()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'int' object has no attribute 'shape'
>>> Xpp
0
>>> Xpp = dfp[numerical_variables].fillna(0)
>>> Xpp
       account_amount_added_12_24m  account_days_in_dc_12_24m  \
89976                            0                        0.0
89977                            0                        0.0
89978                        50956                        0.0
89979                        35054                        0.0
89980                            0                        0.0
89981                            0                        0.0
89982                            0                        0.0
89983                       118135                        0.0
89984                            0                        0.0
89985                         5708                        0.0
89986                            0                        0.0
89987                        12571                        0.0
89988                         8002                        0.0
89989                            0                        0.0
89990                        31175                        0.0
89991                            0                        0.0
89992                        15162                        0.0
89993                            0                        0.0
89994                            0                        0.0
89995                            0                        0.0
89996                            0                        0.0
89997                        50059                        0.0
89998                            0                        0.0
89999                            0                        0.0
90000                            0                        0.0
90001                            0                        0.0
90002                            0                        0.0
90003                         1295                        0.0
90004                         3943                        0.0
90005                            0                        0.0
...                            ...                        ...
99946                        18004                        0.0
99947                            0                        0.0
99948                            0                        0.0
99949                            0                        0.0
99950                            0                        0.0
99951                            0                        0.0
99952                            0                        0.0
99953                        78097                        0.0
99954                        28189                        0.0
99955                            0                        0.0
99956                            0                        0.0
99957                       133654                        0.0
99958                            0                        0.0
99959                            0                        0.0
99960                            0                        0.0
99961                            0                        0.0
99962                            0                        0.0
99963                            0                        0.0
99964                            0                        0.0
99965                       112441                        0.0
99966                        16642                        0.0
99967                        32355                        0.0
99968                            0                        0.0
99969                            0                      365.0
99970                        88405                        0.0
99971                            0                        0.0
99972                            0                        0.0
99973                        45671                        0.0
99974                        56102                        0.0
99975                            0                        0.0

       account_days_in_rem_12_24m  account_days_in_term_12_24m  \
89976                         0.0                          0.0
89977                         0.0                          0.0
89978                        77.0                          0.0
89979                         0.0                          0.0
89980                         0.0                          0.0
89981                         0.0                          0.0
89982                         0.0                          0.0
89983                         0.0                          0.0
89984                         0.0                          0.0
89985                         0.0                          0.0
89986                         0.0                          0.0
89987                         0.0                          0.0
89988                         0.0                          0.0
89989                         0.0                          0.0
89990                        16.0                          0.0
89991                         0.0                          0.0
89992                         0.0                          0.0
89993                         0.0                          0.0
89994                         0.0                          0.0
89995                         0.0                          0.0
89996                         0.0                          0.0
89997                        34.0                          0.0
89998                         0.0                          0.0
89999                         0.0                          0.0
90000                         0.0                          0.0
90001                         0.0                          0.0
90002                         0.0                          0.0
90003                         4.0                          0.0
90004                         0.0                          0.0
90005                         0.0                          0.0
...                           ...                          ...
99946                         0.0                          0.0
99947                         0.0                          0.0
99948                         0.0                          0.0
99949                         0.0                          0.0
99950                         0.0                          0.0
99951                         0.0                          0.0
99952                         0.0                          0.0
99953                        51.0                          0.0
99954                        97.0                          9.0
99955                         0.0                          0.0
99956                         0.0                          0.0
99957                        48.0                          0.0
99958                         0.0                          0.0
99959                         0.0                          0.0
99960                         0.0                          0.0
99961                         0.0                          0.0
99962                         0.0                          0.0
99963                         0.0                          0.0
99964                         0.0                          0.0
99965                        17.0                          0.0
99966                        59.0                          0.0
99967                       125.0                         44.0
99968                         0.0                          0.0
99969                         0.0                          0.0
99970                        15.0                          0.0
99971                         0.0                          0.0
99972                         0.0                          0.0
99973                        20.0                          0.0
99974                         0.0                          0.0
99975                         0.0                          0.0

       account_incoming_debt_vs_paid_0_24m  age  avg_payment_span_0_12m  \
89976                             0.009135   20                6.400000
89977                             0.000000   64                0.000000
89978                             0.000000   28               12.500000
89979                             0.000000   31               41.000000
89980                             0.000000   30                0.000000
89981                             0.000000   22                0.000000
89982                             0.000000   49               19.000000
89983                             0.212976   30               28.500000
89984                             0.000000   31               14.000000
89985                             0.004301   25               17.837209
89986                             0.000000   46               11.375000
89987                             0.264635   30               15.937500
89988                             0.000000   22                0.000000
89989                             0.000011   55                9.555556
89990                             0.759593   34               12.833333
89991                             0.000000   27               22.000000
89992                             1.079245   18               19.000000
89993                             0.000000   56                5.500000
89994                             0.000000   26                7.871795
89995                             0.000000   33                0.000000
89996                             0.000000   31               15.000000
89997                             1.020338   19               19.000000
89998                             0.000000   22                7.000000
89999                             0.000000   49               16.000000
90000                             0.000000   61               43.000000
90001                             0.000000   18               17.428571
90002                             0.000000   36               12.333333
90003                             0.036302   34               25.300000
90004                             0.000000   36               44.400000
90005                             0.000000   22                0.000000
...                                    ...  ...                     ...
99946                             0.000000   21                9.600000
99947                             0.000000   26               12.000000
99948                             0.000000   34               34.000000
99949                             0.000000   49               27.017857
99950                             0.000000   38                5.000000
99951                             0.000000   54               23.777778
99952                             0.000000   34               35.000000
99953                             0.907493   19               18.363636
99954                             0.000438   25               24.800000
99955                             0.992753   20               12.500000
99956                             0.000000   41                6.102564
99957                             0.147310   41               18.289474
99958                             0.000000   18                9.000000
99959                             0.000000   64               14.625000
99960                             0.000000   27               12.000000
99961                             0.000000   35                0.000000
99962                             0.974504   36               14.200000
99963                             0.000000   46                0.000000
99964                             0.000000   51                0.000000
99965                             0.577442   38               13.272727
99966                             0.000000   40               40.000000
99967                             0.665829   28               12.000000
99968                             0.000000   45               17.090909
99969                             0.371604   31                0.000000
99970                             0.672000   21                0.000000
99971                             0.000000   33               10.333333
99972                             0.004044   44               36.000000
99973                             0.705078   24                0.000000
99974                             0.064175   31               17.500000
99975                             0.000000   41               34.666667

       avg_payment_span_0_3m  max_paid_inv_0_12m  max_paid_inv_0_24m  \
89976               5.250000              7225.0              7225.0
89977               0.000000                 0.0                 0.0
89978               0.000000             91980.0             91980.0
89979               0.000000              1790.0              1790.0
89980               0.000000                 0.0                 0.0
89981               0.000000              6055.0              6055.0
89982              19.000000              7985.0              7985.0
89983               7.000000              6560.0              6560.0
89984               0.000000              4790.0             24785.0
89985              16.625000             10290.0             11385.0
89986              11.333333             16790.0             16790.0
89987               4.333333              5380.0              6180.0
89988               0.000000             13885.0             13885.0
89989              10.000000              3990.0             10285.0
89990               0.000000             10209.0             10209.0
89991               0.000000              8286.0              8286.0
89992              19.000000             12650.0             12650.0
89993               0.000000              2675.0              2675.0
89994               4.875000              3244.0              3480.0
89995               0.000000                 0.0                 0.0
89996              15.000000              2500.0              3880.0
89997               0.000000              3290.0              3290.0
89998               0.000000             11220.0             11220.0
89999               0.000000             11325.0             11325.0
90000               0.000000              5583.0              5583.0
90001              16.000000              7690.0              9530.0
90002               0.000000              3680.0              8160.0
90003              32.500000              7750.0             10205.0
90004              26.000000              6143.0             16843.0
90005               0.000000                 0.0             10900.0
...                      ...                 ...                 ...
99946              10.000000              3790.0             13157.0
99947               0.000000               295.0              2042.0
99948               0.000000              4389.0              4389.0
99949              26.666667              7390.0             12990.0
99950               5.000000              1190.0              1190.0
99951              23.500000              7405.0             15340.0
99952              18.000000              4324.0              4324.0
99953               0.000000             10517.0             10517.0
99954               0.000000             11060.0             11060.0
99955              16.000000              9090.0              9090.0
99956               6.666667              6785.0             98385.0
99957              21.833333             13745.0             13745.0
99958               9.000000             11850.0             11850.0
99959              16.500000              3290.0              3290.0
99960              12.000000             14985.0             14985.0
99961               0.000000                 0.0                 0.0
99962               6.500000             11480.0             11480.0
99963               0.000000                 0.0                 0.0
99964               0.000000                 0.0                 0.0
99965              13.666667             14496.0             14496.0
99966              16.000000             11835.0             11835.0
99967               0.000000              9330.0              9330.0
99968              20.714286             12264.0             12264.0
99969               0.000000               895.0               895.0
99970               0.000000                 0.0              6242.0
99971               0.000000             35195.0             35195.0
99972               0.000000              4740.0              4740.0
99973               0.000000              1200.0              1200.0
99974               0.000000             15000.0             15000.0
99975              37.500000             13246.0             14817.0

          ...      num_arch_ok_12_24m  num_arch_rem_0_12m  \
89976     ...                       0                   0
89977     ...                       0                   0
89978     ...                       7                   0
89979     ...                       0                   1
89980     ...                       0                   0
89981     ...                       2                   0
89982     ...                      20                   0
89983     ...                       1                   2
89984     ...                       6                   0
89985     ...                      36                   0
89986     ...                       7                   0
89987     ...                       2                   2
89988     ...                       1                   0
89989     ...                      17                   0
89990     ...                       3                   0
89991     ...                       0                   0
89992     ...                       1                   0
89993     ...                       0                   0
89994     ...                      36                   0
89995     ...                       0                   0
89996     ...                       1                   0
89997     ...                       0                   0
89998     ...                       1                   0
89999     ...                       0                   0
90000     ...                       3                   1
90001     ...                       1                   0
90002     ...                       3                   1
90003     ...                       6                   2
90004     ...                       1                   0
90005     ...                       1                   0
...       ...                     ...                 ...
99946     ...                       5                   0
99947     ...                       1                   0
99948     ...                       2                   1
99949     ...                      94                   1
99950     ...                       0                   0
99951     ...                       3                   4
99952     ...                       0                   2
99953     ...                       4                   3
99954     ...                       3                   1
99955     ...                       0                   0
99956     ...                      19                   0
99957     ...                      29                   1
99958     ...                       0                   0
99959     ...                       9                   0
99960     ...                       2                   0
99961     ...                       0                   0
99962     ...                       1                   0
99963     ...                       0                   0
99964     ...                       0                   0
99965     ...                       5                   0
99966     ...                       5                   5
99967     ...                       0                   0
99968     ...                      44                   0
99969     ...                       0                   0
99970     ...                       2                   0
99971     ...                       2                   0
99972     ...                       3                   0
99973     ...                       0                   0
99974     ...                       1                   0
99975     ...                       2                   1

       num_arch_written_off_0_12m  num_arch_written_off_12_24m  \
89976                         0.0                          0.0
89977                         0.0                          0.0
89978                         0.0                          0.0
89979                         0.0                          0.0
89980                         0.0                          0.0
89981                         0.0                          0.0
89982                         0.0                          0.0
89983                         0.0                          0.0
89984                         0.0                          0.0
89985                         0.0                          0.0
89986                         0.0                          0.0
89987                         0.0                          0.0
89988                         0.0                          0.0
89989                         0.0                          0.0
89990                         0.0                          0.0
89991                         0.0                          0.0
89992                         0.0                          0.0
89993                         0.0                          0.0
89994                         0.0                          0.0
89995                         0.0                          0.0
89996                         0.0                          0.0
89997                         0.0                          0.0
89998                         0.0                          0.0
89999                         0.0                          0.0
90000                         0.0                          0.0
90001                         0.0                          0.0
90002                         0.0                          0.0
90003                         0.0                          0.0
90004                         0.0                          0.0
90005                         0.0                          0.0
...                           ...                          ...
99946                         0.0                          0.0
99947                         0.0                          0.0
99948                         0.0                          0.0
99949                         0.0                          0.0
99950                         0.0                          0.0
99951                         0.0                          0.0
99952                         0.0                          0.0
99953                         0.0                          0.0
99954                         0.0                          0.0
99955                         0.0                          0.0
99956                         0.0                          0.0
99957                         0.0                          0.0
99958                         0.0                          0.0
99959                         0.0                          0.0
99960                         0.0                          0.0
99961                         0.0                          0.0
99962                         0.0                          0.0
99963                         0.0                          0.0
99964                         0.0                          0.0
99965                         0.0                          0.0
99966                         0.0                          0.0
99967                         0.0                          0.0
99968                         0.0                          0.0
99969                         0.0                          0.0
99970                         0.0                          0.0
99971                         0.0                          0.0
99972                         0.0                          0.0
99973                         0.0                          0.0
99974                         0.0                          0.0
99975                         0.0                          0.0

       num_unpaid_bills  recovery_debt  sum_capital_paid_account_0_12m  \
89976                 1              0                            8815
89977                 0              0                               0
89978                 0              0                           36163
89979                 0              0                           62585
89980                 0              0                           14295
89981                 0              0                               0
89982                 0              0                               0
89983                12              0                           37657
89984                 0              0                               0
89985                 3              0                            5707
89986                 2              0                               0
89987                 2              0                          121531
89988                 1              0                               0
89989                 1              0                           89579
89990                12              0                           30393
89991                 0              0                               0
89992                 1              0                               0
89993                 2              0                            4645
89994                 1              0                               0
89995                 0              0                               0
89996                 0              0                               0
89997                 5              0                           25102
89998                 2              0                               0
89999                 0              0                               0
90000                 2              0                               0
90001                 1              0                               0
90002                 1              0                               0
90003                 3              0                               0
90004                 1              0                               0
90005                 0              0                               0
...                 ...            ...                             ...
99946                 0              0                               0
99947                 0              0                               0
99948                 0              0                               0
99949                 2              0                               0
99950                 0              0                               0
99951                 2              0                               0
99952                 1              0                               0
99953                22              0                           45258
99954                 1              0                           18585
99955                 3              0                           21385
99956                 1              0                               0
99957                17              0                          190459
99958                 0              0                               0
99959                 1              0                               0
99960                 0              0                               0
99961                 0              0                               0
99962                 1              0                            6075
99963                 0              0                               0
99964                 0              0                               0
99965                16              0                           74306
99966                 2              0                           12738
99967                16              0                           20633
99968                 3              0                               0
99969                 2              0                            7695
99970                 9              0                           28870
99971                 0              0                               0
99972                 1              0                            7948
99973                18              0                           17447
99974                 1              0                           18339
99975                 1              0                               0

       sum_capital_paid_account_12_24m  sum_paid_inv_0_12m  time_hours
89976                                0               27157   19.895556
89977                                0                   0    0.236667
89978                            39846               93760   20.332778
89979                                0                1790    6.201111
89980                                0                   0    8.451111
89981                                0                6055   22.263056
89982                                0               64885   14.909444
89983                            42836               13483    4.866111
89984                                0                4790    9.260000
89985                                0              131364   12.881944
89986                                0               91572   22.881667
89987                            12480               59075   16.904444
89988                                0               20080   13.603611
89989                                0               20330   12.230833
89990                            21958               28224    9.644444
89991                                0                8286   22.792222
89992                                0               12650   10.028333
89993                                0                6155   16.773056
89994                                0               51025   20.314722
89995                                0                   0   12.371111
89996                                0                6400   15.847778
89997                                0                4880   13.968611
89998                                0               28520   15.454167
89999                                0               16095    9.953333
90000                                0                9906   21.117222
90001                                0               34485   10.009722
90002                                0               11805   22.726667
90003                            11997               39960   19.816389
90004                             5400               26020   19.588611
90005                                0                   0   16.371389
...                                ...                 ...         ...
99946                                0               15755   11.413889
99947                                0                 295   13.463889
99948                                0                4889   14.484167
99949                                0              136892   12.618056
99950                                0                1190   19.784722
99951                                0               39835    9.177500
99952                                0               12084   13.224167
99953                            18905               50092   21.173056
99954                             9669               25080   21.589444
99955                                0               22390   13.678333
99956                                0              128790    8.074167
99957                           145771              100650    7.998056
99958                                0               11850   18.946389
99959                                0               19165   22.544167
99960                                0               58008   15.111389
99961                                0                   0   15.560000
99962                                0               21110   18.788611
99963                                0                   0   10.114444
99964                                0                   0   12.478889
99965                            30838               97672   14.867778
99966                            10852               44244   17.471389
99967                            22355               10225   13.057500
99968                                0              276135   19.786944
99969                             1025                 895   11.290833
99970                            25771                   0    9.060833
99971                                0               60127   10.765556
99972                                0                4740   21.708333
99973                            19627                3100    2.185278
99974                            56180               34785    9.725278
99975                                0               30602   11.585278

[10000 rows x 25 columns]
>>> Xpp.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 89976 to 99975
Data columns (total 25 columns):
account_amount_added_12_24m            10000 non-null int64
account_days_in_dc_12_24m              10000 non-null float64
account_days_in_rem_12_24m             10000 non-null float64
account_days_in_term_12_24m            10000 non-null float64
account_incoming_debt_vs_paid_0_24m    10000 non-null float64
age                                    10000 non-null int64
avg_payment_span_0_12m                 10000 non-null float64
avg_payment_span_0_3m                  10000 non-null float64
max_paid_inv_0_12m                     10000 non-null float64
max_paid_inv_0_24m                     10000 non-null float64
num_active_div_by_paid_inv_0_12m       10000 non-null float64
num_active_inv                         10000 non-null int64
num_arch_dc_0_12m                      10000 non-null int64
num_arch_dc_12_24m                     10000 non-null int64
num_arch_ok_0_12m                      10000 non-null int64
num_arch_ok_12_24m                     10000 non-null int64
num_arch_rem_0_12m                     10000 non-null int64
num_arch_written_off_0_12m             10000 non-null float64
num_arch_written_off_12_24m            10000 non-null float64
num_unpaid_bills                       10000 non-null int64
recovery_debt                          10000 non-null int64
sum_capital_paid_account_0_12m         10000 non-null int64
sum_capital_paid_account_12_24m        10000 non-null int64
sum_paid_inv_0_12m                     10000 non-null int64
time_hours                             10000 non-null float64
dtypes: float64(12), int64(13)
memory usage: 2.0 MB
>>> Xpp = dfp[numerical_variables].fillna(0).as_matrix()
>>> Xpp
array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   2.71570000e+04,   1.98955556e+01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   0.00000000e+00,   2.36666667e-01],
       [  5.09560000e+04,   0.00000000e+00,   7.70000000e+01, ...,
          3.98460000e+04,   9.37600000e+04,   2.03327778e+01],
       ...,
       [  4.56710000e+04,   0.00000000e+00,   2.00000000e+01, ...,
          1.96270000e+04,   3.10000000e+03,   2.18527778e+00],
       [  5.61020000e+04,   0.00000000e+00,   0.00000000e+00, ...,
          5.61800000e+04,   3.47850000e+04,   9.72527778e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
          0.00000000e+00,   3.06020000e+04,   1.15852778e+01]])
>>> (Xpp != Xp).sum()
22535
>>> yp = knn.predict(Xpp)
>>> yp
array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
>>> yp.sum()
0.0
>>> yp.sum() == 0
True
>>>
>>>
>>> logreg = LogisticRegression()
>>> logreg.fit(X_train, Y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
>>>
>>> # STEP 3: make predictions on the testing set
... y_hat = logreg.predict(X_test)
>>>
>>> #compare actual response values (y_test) with predicted response values (y_hat)
... print(metrics.accuracy_score(Y_test,y_hat))
0.989025460931
>>>
>>> yp_logreg = logreg.predict(Xpp)
>>> y_hat.sum()
3.0
>>> yp_logreg.sum()
561.0
>>> len(yp_logreg)
10000
>>> yp_logreg.sum() / len(yp_logreg), dfm['default'].mean()
(0.056099999999999997, 0.009987926682032707)
>>> yp_logreg = logreg.predict(Xpp)
>>> yp_logreg = logreg.predict(Xp)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/base.py", line 324, in predict
    scores = self.decision_function(X)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/base.py", line 300, in decision_function
    X = check_array(X, accept_sparse='csr')
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 453, in check_array
    _assert_all_finite(array)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 44, in _assert_all_finite
    " or a value too large for %r." % X.dtype)
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
>>> Xpp.shape()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object is not callable
>>> Xpp.shape
(10000, 25)
>>> Xpp.shape == Xp.shape
True
>>> (Xpp != Xp).sum() > 0 #True
True
>>> yp.sum() #No of predicted defaults
0.0
>>>
>>> logreg = LogisticRegression()
>>> logreg.fit(X_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
>>>
>>> # STEP 3: make predictions on the testing set
... y_hat_reg = logreg.predict(X_test)
>>> y_hat_reg.sum()
3.0
>>> #compare actual response values (y_test) with predicted response values (y_hat)
... print(metrics.accuracy_score(Y_test,y_hat_reg))
0.989025460931
>>>
>>> y_p_reg = logreg.predict(Xpp)
>>> # compare % predicted defaults vs % default in dataset
... y_p_reg.sum() / len(y_p_reg), dfm['default'].mean()
(0.056099999999999997, 0.009987926682032707)
>>> (y_p_reg.sum() / len(y_p_reg), dfm['default'].mean())*100
(0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707)
>>> (y_p_reg.sum() / len(y_p_reg), dfm['default'].mean())
(0.056099999999999997, 0.009987926682032707)
>>> 100 * (y_p_reg.sum() / len(y_p_reg), dfm['default'].mean())
(0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707, 0.056099999999999997, 0.009987926682032707)
>>> (y_p_reg.sum() / len(y_p_reg), dfm['default'].mean())
(0.056099999999999997, 0.009987926682032707)
>>> y_p_reg.sum() / len(y_p_reg), dfm['default'].mean()
(0.056099999999999997, 0.009987926682032707)
>>>
>>>
>>> y_p_reg.sum()
561.0
>>> y_o_knn.sum()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_o_knn' is not defined
>>> y_p_knn.sum()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_p_knn' is not defined
>>> y_p_knn.sum() #No of predicted defaults
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_p_knn' is not defined
>>>
>>>
>>>
>>> knn = KNeighborsClassifier(n_neighbors=5)
>>> knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
>>> y_hat_knn = knn.predict(X_test)
>>> print(y_hat.sum(), len(y_hat))
3.0 2278
>>> print(metrics.accuracy_score(y_test, y_hat_knn))
0.990342405619
>>>
>>> y_p_knn_error = knn.predict(Xp) # qq vrf fuckar det up.
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/neighbors/classification.py", line 143, in predict
    X = check_array(X, accept_sparse='csr')
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 453, in check_array
    _assert_all_finite(array)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 44, in _assert_all_finite
    " or a value too large for %r." % X.dtype)
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
>>> dfp.drop('default', axis=1).fillna(0).isnull().sum().sum()
0
>>> dfp[numerical_variables].fillna(0).info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 10000 entries, 89976 to 99975
Data columns (total 25 columns):
account_amount_added_12_24m            10000 non-null int64
account_days_in_dc_12_24m              10000 non-null float64
account_days_in_rem_12_24m             10000 non-null float64
account_days_in_term_12_24m            10000 non-null float64
account_incoming_debt_vs_paid_0_24m    10000 non-null float64
age                                    10000 non-null int64
avg_payment_span_0_12m                 10000 non-null float64
avg_payment_span_0_3m                  10000 non-null float64
max_paid_inv_0_12m                     10000 non-null float64
max_paid_inv_0_24m                     10000 non-null float64
num_active_div_by_paid_inv_0_12m       10000 non-null float64
num_active_inv                         10000 non-null int64
num_arch_dc_0_12m                      10000 non-null int64
num_arch_dc_12_24m                     10000 non-null int64
num_arch_ok_0_12m                      10000 non-null int64
num_arch_ok_12_24m                     10000 non-null int64
num_arch_rem_0_12m                     10000 non-null int64
num_arch_written_off_0_12m             10000 non-null float64
num_arch_written_off_12_24m            10000 non-null float64
num_unpaid_bills                       10000 non-null int64
recovery_debt                          10000 non-null int64
sum_capital_paid_account_0_12m         10000 non-null int64
sum_capital_paid_account_12_24m        10000 non-null int64
sum_paid_inv_0_12m                     10000 non-null int64
time_hours                             10000 non-null float64
dtypes: float64(12), int64(13)
memory usage: 2.0 MB
>>> Xpp = dfp[numerical_variables].fillna(0).as_matrix()
>>> (Xpp != Xp).sum() > 0 #True
True
>>> Xpp.shape == Xp.shape #True
True
>>> y_p_knn = knn.predict(Xpp)
>>> y_p_knn.sum() #No of predicted defaults
0.0
>>>
>>> dfm.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 9111 entries, 7 to 89966
Data columns (total 43 columns):
uuid                                   9111 non-null object
default                                9111 non-null float64
account_amount_added_12_24m            9111 non-null int64
account_days_in_dc_12_24m              9111 non-null float64
account_days_in_rem_12_24m             9111 non-null float64
account_days_in_term_12_24m            9111 non-null float64
account_incoming_debt_vs_paid_0_24m    9111 non-null float64
account_status                         9111 non-null float64
account_worst_status_0_3m              9111 non-null float64
account_worst_status_12_24m            9111 non-null float64
account_worst_status_3_6m              9111 non-null float64
account_worst_status_6_12m             9111 non-null float64
age                                    9111 non-null int64
avg_payment_span_0_12m                 9111 non-null float64
avg_payment_span_0_3m                  9111 non-null float64
merchant_category                      9111 non-null object
merchant_group                         9111 non-null object
has_paid                               9111 non-null bool
max_paid_inv_0_12m                     9111 non-null float64
max_paid_inv_0_24m                     9111 non-null float64
name_in_email                          9111 non-null object
num_active_div_by_paid_inv_0_12m       9111 non-null float64
num_active_inv                         9111 non-null int64
num_arch_dc_0_12m                      9111 non-null int64
num_arch_dc_12_24m                     9111 non-null int64
num_arch_ok_0_12m                      9111 non-null int64
num_arch_ok_12_24m                     9111 non-null int64
num_arch_rem_0_12m                     9111 non-null int64
num_arch_written_off_0_12m             9111 non-null float64
num_arch_written_off_12_24m            9111 non-null float64
num_unpaid_bills                       9111 non-null int64
status_last_archived_0_24m             9111 non-null int64
status_2nd_last_archived_0_24m         9111 non-null int64
status_3rd_last_archived_0_24m         9111 non-null int64
status_max_archived_0_6_months         9111 non-null int64
status_max_archived_0_12_months        9111 non-null int64
status_max_archived_0_24_months        9111 non-null int64
recovery_debt                          9111 non-null int64
sum_capital_paid_account_0_12m         9111 non-null int64
sum_capital_paid_account_12_24m        9111 non-null int64
sum_paid_inv_0_12m                     9111 non-null int64
time_hours                             9111 non-null float64
worst_status_active_inv                9111 non-null float64
dtypes: bool(1), float64(19), int64(19), object(4)
memory usage: 3.3+ MB
>>> raw_data[pd.notnull(raw_data.default)].info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 89976 entries, 0 to 89975
Data columns (total 43 columns):
uuid                                   89976 non-null object
default                                89976 non-null float64
account_amount_added_12_24m            89976 non-null int64
account_days_in_dc_12_24m              79293 non-null float64
account_days_in_rem_12_24m             79293 non-null float64
account_days_in_term_12_24m            79293 non-null float64
account_incoming_debt_vs_paid_0_24m    36619 non-null float64
account_status                         41042 non-null float64
account_worst_status_0_3m              41042 non-null float64
account_worst_status_12_24m            29921 non-null float64
account_worst_status_3_6m              38038 non-null float64
account_worst_status_6_12m             35663 non-null float64
age                                    89976 non-null int64
avg_payment_span_0_12m                 68508 non-null float64
avg_payment_span_0_3m                  45594 non-null float64
merchant_category                      89976 non-null object
merchant_group                         89976 non-null object
has_paid                               89976 non-null bool
max_paid_inv_0_12m                     89976 non-null float64
max_paid_inv_0_24m                     89976 non-null float64
name_in_email                          89976 non-null object
num_active_div_by_paid_inv_0_12m       69318 non-null float64
num_active_inv                         89976 non-null int64
num_arch_dc_0_12m                      89976 non-null int64
num_arch_dc_12_24m                     89976 non-null int64
num_arch_ok_0_12m                      89976 non-null int64
num_arch_ok_12_24m                     89976 non-null int64
num_arch_rem_0_12m                     89976 non-null int64
num_arch_written_off_0_12m             73671 non-null float64
num_arch_written_off_12_24m            73671 non-null float64
num_unpaid_bills                       89976 non-null int64
status_last_archived_0_24m             89976 non-null int64
status_2nd_last_archived_0_24m         89976 non-null int64
status_3rd_last_archived_0_24m         89976 non-null int64
status_max_archived_0_6_months         89976 non-null int64
status_max_archived_0_12_months        89976 non-null int64
status_max_archived_0_24_months        89976 non-null int64
recovery_debt                          89976 non-null int64
sum_capital_paid_account_0_12m         89976 non-null int64
sum_capital_paid_account_12_24m        89976 non-null int64
sum_paid_inv_0_12m                     89976 non-null int64
time_hours                             89976 non-null float64
worst_status_active_inv                27436 non-null float64
dtypes: bool(1), float64(19), int64(19), object(4)
memory usage: 29.6+ MB
>>> raw_data[pd.notnull(raw_data.default)].isnull()
        uuid  default  account_amount_added_12_24m  account_days_in_dc_12_24m  \
0      False    False                        False                      False
1      False    False                        False                      False
2      False    False                        False                      False
3      False    False                        False                       True
4      False    False                        False                      False
5      False    False                        False                      False
6      False    False                        False                      False
7      False    False                        False                      False
8      False    False                        False                      False
9      False    False                        False                      False
10     False    False                        False                      False
11     False    False                        False                       True
12     False    False                        False                      False
13     False    False                        False                      False
14     False    False                        False                      False
15     False    False                        False                      False
16     False    False                        False                      False
17     False    False                        False                       True
18     False    False                        False                      False
19     False    False                        False                      False
20     False    False                        False                      False
21     False    False                        False                      False
22     False    False                        False                       True
23     False    False                        False                      False
24     False    False                        False                      False
25     False    False                        False                      False
26     False    False                        False                      False
27     False    False                        False                      False
28     False    False                        False                      False
29     False    False                        False                      False
...      ...      ...                          ...                        ...
89946  False    False                        False                       True
89947  False    False                        False                       True
89948  False    False                        False                       True
89949  False    False                        False                       True
89950  False    False                        False                      False
89951  False    False                        False                      False
89952  False    False                        False                      False
89953  False    False                        False                      False
89954  False    False                        False                       True
89955  False    False                        False                      False
89956  False    False                        False                      False
89957  False    False                        False                      False
89958  False    False                        False                      False
89959  False    False                        False                      False
89960  False    False                        False                      False
89961  False    False                        False                      False
89962  False    False                        False                      False
89963  False    False                        False                      False
89964  False    False                        False                      False
89965  False    False                        False                      False
89966  False    False                        False                      False
89967  False    False                        False                      False
89968  False    False                        False                       True
89969  False    False                        False                      False
89970  False    False                        False                      False
89971  False    False                        False                      False
89972  False    False                        False                      False
89973  False    False                        False                      False
89974  False    False                        False                      False
89975  False    False                        False                       True

       account_days_in_rem_12_24m  account_days_in_term_12_24m  \
0                           False                        False
1                           False                        False
2                           False                        False
3                            True                         True
4                           False                        False
5                           False                        False
6                           False                        False
7                           False                        False
8                           False                        False
9                           False                        False
10                          False                        False
11                           True                         True
12                          False                        False
13                          False                        False
14                          False                        False
15                          False                        False
16                          False                        False
17                           True                         True
18                          False                        False
19                          False                        False
20                          False                        False
21                          False                        False
22                           True                         True
23                          False                        False
24                          False                        False
25                          False                        False
26                          False                        False
27                          False                        False
28                          False                        False
29                          False                        False
...                           ...                          ...
89946                        True                         True
89947                        True                         True
89948                        True                         True
89949                        True                         True
89950                       False                        False
89951                       False                        False
89952                       False                        False
89953                       False                        False
89954                        True                         True
89955                       False                        False
89956                       False                        False
89957                       False                        False
89958                       False                        False
89959                       False                        False
89960                       False                        False
89961                       False                        False
89962                       False                        False
89963                       False                        False
89964                       False                        False
89965                       False                        False
89966                       False                        False
89967                       False                        False
89968                        True                         True
89969                       False                        False
89970                       False                        False
89971                       False                        False
89972                       False                        False
89973                       False                        False
89974                       False                        False
89975                        True                         True

       account_incoming_debt_vs_paid_0_24m  account_status  \
0                                    False           False
1                                     True           False
2                                     True            True
3                                     True            True
4                                     True            True
5                                     True            True
6                                    False           False
7                                    False           False
8                                    False           False
9                                     True            True
10                                    True            True
11                                    True            True
12                                   False           False
13                                    True            True
14                                    True            True
15                                    True            True
16                                    True            True
17                                    True            True
18                                    True            True
19                                    True            True
20                                   False           False
21                                   False           False
22                                    True            True
23                                    True            True
24                                    True            True
25                                    True            True
26                                   False           False
27                                   False           False
28                                    True            True
29                                    True            True
...                                    ...             ...
89946                                 True            True
89947                                 True            True
89948                                 True            True
89949                                 True            True
89950                                False           False
89951                                False           False
89952                                 True            True
89953                                False           False
89954                                 True            True
89955                                False           False
89956                                 True            True
89957                                 True           False
89958                                 True            True
89959                                 True            True
89960                                 True            True
89961                                False           False
89962                                 True            True
89963                                False           False
89964                                 True            True
89965                                False           False
89966                                False           False
89967                                 True            True
89968                                 True            True
89969                                False           False
89970                                False           False
89971                                 True            True
89972                                 True            True
89973                                 True           False
89974                                 True            True
89975                                 True            True

       account_worst_status_0_3m  account_worst_status_12_24m  \
0                          False                         True
1                          False                        False
2                           True                         True
3                           True                         True
4                           True                         True
5                           True                         True
6                          False                        False
7                          False                        False
8                          False                        False
9                           True                         True
10                          True                         True
11                          True                         True
12                         False                        False
13                          True                         True
14                          True                         True
15                          True                         True
16                          True                         True
17                          True                         True
18                          True                         True
19                          True                         True
20                         False                         True
21                         False                        False
22                          True                         True
23                          True                         True
24                          True                         True
25                          True                         True
26                         False                         True
27                         False                        False
28                          True                         True
29                          True                         True
...                          ...                          ...
89946                       True                         True
89947                       True                         True
89948                       True                         True
89949                       True                         True
89950                      False                        False
89951                      False                        False
89952                       True                         True
89953                      False                        False
89954                       True                         True
89955                      False                        False
89956                       True                         True
89957                      False                        False
89958                       True                         True
89959                       True                         True
89960                       True                         True
89961                      False                        False
89962                       True                         True
89963                      False                        False
89964                       True                         True
89965                      False                        False
89966                      False                        False
89967                       True                         True
89968                       True                         True
89969                      False                        False
89970                      False                         True
89971                       True                         True
89972                       True                         True
89973                      False                        False
89974                       True                         True
89975                       True                         True

                ...             status_3rd_last_archived_0_24m  \
0               ...                                      False
1               ...                                      False
2               ...                                      False
3               ...                                      False
4               ...                                      False
5               ...                                      False
6               ...                                      False
7               ...                                      False
8               ...                                      False
9               ...                                      False
10              ...                                      False
11              ...                                      False
12              ...                                      False
13              ...                                      False
14              ...                                      False
15              ...                                      False
16              ...                                      False
17              ...                                      False
18              ...                                      False
19              ...                                      False
20              ...                                      False
21              ...                                      False
22              ...                                      False
23              ...                                      False
24              ...                                      False
25              ...                                      False
26              ...                                      False
27              ...                                      False
28              ...                                      False
29              ...                                      False
...             ...                                        ...
89946           ...                                      False
89947           ...                                      False
89948           ...                                      False
89949           ...                                      False
89950           ...                                      False
89951           ...                                      False
89952           ...                                      False
89953           ...                                      False
89954           ...                                      False
89955           ...                                      False
89956           ...                                      False
89957           ...                                      False
89958           ...                                      False
89959           ...                                      False
89960           ...                                      False
89961           ...                                      False
89962           ...                                      False
89963           ...                                      False
89964           ...                                      False
89965           ...                                      False
89966           ...                                      False
89967           ...                                      False
89968           ...                                      False
89969           ...                                      False
89970           ...                                      False
89971           ...                                      False
89972           ...                                      False
89973           ...                                      False
89974           ...                                      False
89975           ...                                      False

       status_max_archived_0_6_months  status_max_archived_0_12_months  \
0                               False                            False
1                               False                            False
2                               False                            False
3                               False                            False
4                               False                            False
5                               False                            False
6                               False                            False
7                               False                            False
8                               False                            False
9                               False                            False
10                              False                            False
11                              False                            False
12                              False                            False
13                              False                            False
14                              False                            False
15                              False                            False
16                              False                            False
17                              False                            False
18                              False                            False
19                              False                            False
20                              False                            False
21                              False                            False
22                              False                            False
23                              False                            False
24                              False                            False
25                              False                            False
26                              False                            False
27                              False                            False
28                              False                            False
29                              False                            False
...                               ...                              ...
89946                           False                            False
89947                           False                            False
89948                           False                            False
89949                           False                            False
89950                           False                            False
89951                           False                            False
89952                           False                            False
89953                           False                            False
89954                           False                            False
89955                           False                            False
89956                           False                            False
89957                           False                            False
89958                           False                            False
89959                           False                            False
89960                           False                            False
89961                           False                            False
89962                           False                            False
89963                           False                            False
89964                           False                            False
89965                           False                            False
89966                           False                            False
89967                           False                            False
89968                           False                            False
89969                           False                            False
89970                           False                            False
89971                           False                            False
89972                           False                            False
89973                           False                            False
89974                           False                            False
89975                           False                            False

       status_max_archived_0_24_months  recovery_debt  \
0                                False          False
1                                False          False
2                                False          False
3                                False          False
4                                False          False
5                                False          False
6                                False          False
7                                False          False
8                                False          False
9                                False          False
10                               False          False
11                               False          False
12                               False          False
13                               False          False
14                               False          False
15                               False          False
16                               False          False
17                               False          False
18                               False          False
19                               False          False
20                               False          False
21                               False          False
22                               False          False
23                               False          False
24                               False          False
25                               False          False
26                               False          False
27                               False          False
28                               False          False
29                               False          False
...                                ...            ...
89946                            False          False
89947                            False          False
89948                            False          False
89949                            False          False
89950                            False          False
89951                            False          False
89952                            False          False
89953                            False          False
89954                            False          False
89955                            False          False
89956                            False          False
89957                            False          False
89958                            False          False
89959                            False          False
89960                            False          False
89961                            False          False
89962                            False          False
89963                            False          False
89964                            False          False
89965                            False          False
89966                            False          False
89967                            False          False
89968                            False          False
89969                            False          False
89970                            False          False
89971                            False          False
89972                            False          False
89973                            False          False
89974                            False          False
89975                            False          False

       sum_capital_paid_account_0_12m  sum_capital_paid_account_12_24m  \
0                               False                            False
1                               False                            False
2                               False                            False
3                               False                            False
4                               False                            False
5                               False                            False
6                               False                            False
7                               False                            False
8                               False                            False
9                               False                            False
10                              False                            False
11                              False                            False
12                              False                            False
13                              False                            False
14                              False                            False
15                              False                            False
16                              False                            False
17                              False                            False
18                              False                            False
19                              False                            False
20                              False                            False
21                              False                            False
22                              False                            False
23                              False                            False
24                              False                            False
25                              False                            False
26                              False                            False
27                              False                            False
28                              False                            False
29                              False                            False
...                               ...                              ...
89946                           False                            False
89947                           False                            False
89948                           False                            False
89949                           False                            False
89950                           False                            False
89951                           False                            False
89952                           False                            False
89953                           False                            False
89954                           False                            False
89955                           False                            False
89956                           False                            False
89957                           False                            False
89958                           False                            False
89959                           False                            False
89960                           False                            False
89961                           False                            False
89962                           False                            False
89963                           False                            False
89964                           False                            False
89965                           False                            False
89966                           False                            False
89967                           False                            False
89968                           False                            False
89969                           False                            False
89970                           False                            False
89971                           False                            False
89972                           False                            False
89973                           False                            False
89974                           False                            False
89975                           False                            False

       sum_paid_inv_0_12m  time_hours  worst_status_active_inv
0                   False       False                    False
1                   False       False                     True
2                   False       False                    False
3                   False       False                    False
4                   False       False                     True
5                   False       False                     True
6                   False       False                     True
7                   False       False                    False
8                   False       False                    False
9                   False       False                     True
10                  False       False                     True
11                  False       False                     True
12                  False       False                     True
13                  False       False                     True
14                  False       False                     True
15                  False       False                     True
16                  False       False                     True
17                  False       False                     True
18                  False       False                     True
19                  False       False                     True
20                  False       False                    False
21                  False       False                     True
22                  False       False                    False
23                  False       False                     True
24                  False       False                     True
25                  False       False                     True
26                  False       False                    False
27                  False       False                     True
28                  False       False                     True
29                  False       False                     True
...                   ...         ...                      ...
89946               False       False                    False
89947               False       False                     True
89948               False       False                     True
89949               False       False                     True
89950               False       False                    False
89951               False       False                     True
89952               False       False                     True
89953               False       False                     True
89954               False       False                     True
89955               False       False                     True
89956               False       False                     True
89957               False       False                     True
89958               False       False                     True
89959               False       False                    False
89960               False       False                     True
89961               False       False                     True
89962               False       False                     True
89963               False       False                     True
89964               False       False                    False
89965               False       False                     True
89966               False       False                    False
89967               False       False                     True
89968               False       False                     True
89969               False       False                     True
89970               False       False                     True
89971               False       False                     True
89972               False       False                     True
89973               False       False                     True
89974               False       False                     True
89975               False       False                     True

[89976 rows x 43 columns]
>>> raw_data[pd.notnull(raw_data.default)].isnull().sum()
uuid                                       0
default                                    0
account_amount_added_12_24m                0
account_days_in_dc_12_24m              10683
account_days_in_rem_12_24m             10683
account_days_in_term_12_24m            10683
account_incoming_debt_vs_paid_0_24m    53357
account_status                         48934
account_worst_status_0_3m              48934
account_worst_status_12_24m            60055
account_worst_status_3_6m              51938
account_worst_status_6_12m             54313
age                                        0
avg_payment_span_0_12m                 21468
avg_payment_span_0_3m                  44382
merchant_category                          0
merchant_group                             0
has_paid                                   0
max_paid_inv_0_12m                         0
max_paid_inv_0_24m                         0
name_in_email                              0
num_active_div_by_paid_inv_0_12m       20658
num_active_inv                             0
num_arch_dc_0_12m                          0
num_arch_dc_12_24m                         0
num_arch_ok_0_12m                          0
num_arch_ok_12_24m                         0
num_arch_rem_0_12m                         0
num_arch_written_off_0_12m             16305
num_arch_written_off_12_24m            16305
num_unpaid_bills                           0
status_last_archived_0_24m                 0
status_2nd_last_archived_0_24m             0
status_3rd_last_archived_0_24m             0
status_max_archived_0_6_months             0
status_max_archived_0_12_months            0
status_max_archived_0_24_months            0
recovery_debt                              0
sum_capital_paid_account_0_12m             0
sum_capital_paid_account_12_24m            0
sum_paid_inv_0_12m                         0
time_hours                                 0
worst_status_active_inv                62540
dtype: int64
>>> dfp.drop('default', axis=1).fillna(0).isnull().sum().sum()
0
>>> import keras
Using TensorFlow backend.
2017-11-19 19:20:17.548419: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-19 19:20:17.548491: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-11-19 19:20:17.548501: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
>>> from keras.layers import Dense
>>> from keras.models import Sequential
>>> from keras.utils import to_categorical
>>>
>>> # Specify
... model = Sequential()
>>> n_cols = X_train.shape[1]
>>> shape = (n_cols,)
>>> np.round(n_cols * 2/3, 0) # nodes in layer 1
17.0
>>> np.round(nodes_1 / 2, 0) # nodes i layer 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'nodes_1' is not defined
>>> model.add(Dense(17, activation='relu', input_shape=shape))
>>> #model.add(Dense(8, activation='relu', input_shape=shape))
... model.add(Dense(1, activation='sigmoid'))
>>> model.compile(optimizer='adam',
...               loss='binary_crossentropy',
...               metrics=['accuracy'])
>>>
>>>
>>> # Fit
... model.fit(X_train, y_train, epochs = 5)
Epoch 1/5
  32/6833 [..............................] - ETA: 2:51 - loss: 1.1570 -  352/6833 [>.............................] - ETA: 15s - loss: 1.2823 - a1184/6833 [====>.........................] - ETA: 4s - loss: 1.1439 - ac1952/6833 [=======>......................] - ETA: 2s - loss: 1.0254 - ac2944/6833 [===========>..................] - ETA: 1s - loss: 0.8940 - ac3744/6833 [===============>..............] - ETA: 0s - loss: 0.8125 - ac4736/6833 [===================>..........] - ETA: 0s - loss: 0.7307 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.6711 - ac6496/6833 [===========================>..] - ETA: 0s - loss: 0.6220 - ac6833/6833 [==============================] - 1s 181us/step - loss: 0.6037 - acc: 0.7098
Epoch 2/5
  32/6833 [..............................] - ETA: 0s - loss: 0.2967 - ac 864/6833 [==>...........................] - ETA: 0s - loss: 0.2380 - ac1760/6833 [======>.......................] - ETA: 0s - loss: 0.2228 - ac2688/6833 [==========>...................] - ETA: 0s - loss: 0.2038 - ac3616/6833 [==============>...............] - ETA: 0s - loss: 0.1896 - ac4608/6833 [===================>..........] - ETA: 0s - loss: 0.1801 - ac5472/6833 [=======================>......] - ETA: 0s - loss: 0.1728 - ac6432/6833 [===========================>..] - ETA: 0s - loss: 0.1638 - ac6833/6833 [==============================] - 0s 56us/step - loss: 0.1588 - acc: 0.9889
Epoch 3/5
  32/6833 [..............................] - ETA: 0s - loss: 0.1469 - ac 928/6833 [===>..........................] - ETA: 0s - loss: 0.0947 - ac1888/6833 [=======>......................] - ETA: 0s - loss: 0.0982 - ac2624/6833 [==========>...................] - ETA: 0s - loss: 0.0983 - ac3424/6833 [==============>...............] - ETA: 0s - loss: 0.0933 - ac4384/6833 [==================>...........] - ETA: 0s - loss: 0.0893 - ac5376/6833 [======================>.......] - ETA: 0s - loss: 0.0889 - ac6176/6833 [==========================>...] - ETA: 0s - loss: 0.0837 - ac6833/6833 [==============================] - 0s 58us/step - loss: 0.0810 - acc: 0.9895
Epoch 4/5
  32/6833 [..............................] - ETA: 0s - loss: 0.0527 - ac 992/6833 [===>..........................] - ETA: 0s - loss: 0.0590 - ac1952/6833 [=======>......................] - ETA: 0s - loss: 0.0570 - ac2848/6833 [===========>..................] - ETA: 0s - loss: 0.0582 - ac3584/6833 [==============>...............] - ETA: 0s - loss: 0.0633 - ac4480/6833 [==================>...........] - ETA: 0s - loss: 0.0639 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0606 - ac6304/6833 [==========================>...] - ETA: 0s - loss: 0.0593 - ac6833/6833 [==============================] - 0s 58us/step - loss: 0.0599 - acc: 0.9900
Epoch 5/5
  32/6833 [..............................] - ETA: 0s - loss: 0.0250 - ac 992/6833 [===>..........................] - ETA: 0s - loss: 0.0432 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0420 - ac2720/6833 [==========>...................] - ETA: 0s - loss: 0.0442 - ac3744/6833 [===============>..............] - ETA: 0s - loss: 0.0472 - ac4672/6833 [===================>..........] - ETA: 0s - loss: 0.0499 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0503 - ac6464/6833 [===========================>..] - ETA: 0s - loss: 0.0507 - ac6833/6833 [==============================] - 0s 57us/step - loss: 0.0511 - acc: 0.9902
<keras.callbacks.History object at 0x7fbdb8bfd9e8>
>>> early_stopping_monitor = EarlyStopping(patience = 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'EarlyStopping' is not defined
>>>
>>> model.fit(X_train, y_train, epochs = 25,
...           batch_size = 64,
...           callbacks=[early_stopping_monitor])
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
NameError: name 'early_stopping_monitor' is not defined
>>>
>>> early_stopping_monitor = EarlyStopping(patience = 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'EarlyStopping' is not defined
>>> from keras.callbacks import EarlyStopping
>>> early_stopping_monitor = EarlyStopping(patience = 3)
>>>
>>> model.fit(X_train, y_train, epochs = 25,
...           batch_size = 64,
...           callbacks=[early_stopping_monitor])
Epoch 1/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0173 - ac1536/6833 [=====>........................] - ETA: 0s - loss: 0.0442 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0467 - ac4992/6833 [====================>.........] - ETA: 0s - loss: 0.0464 - ac6592/6833 [===========================>..] - ETA: 0s - loss: 0.0470 - acc: 0.9906/home/jl/anaconda3/lib/python3.6/site-packages/keras/callbacks.py:493: RuntimeWarning: Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,acc
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
6833/6833 [==============================] - 0s 42us/step - loss: 0.0469 - acc: 0.9906
Epoch 2/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0833 - ac1600/6833 [======>.......................] - ETA: 0s - loss: 0.0448 - ac3136/6833 [============>.................] - ETA: 0s - loss: 0.0468 - ac5184/6833 [=====================>........] - ETA: 0s - loss: 0.0455 - ac6720/6833 [============================>.] - ETA: 0s - loss: 0.0456 - ac6833/6833 [==============================] - 0s 31us/step - loss: 0.0452 - acc: 0.9906
Epoch 3/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0122 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0441 - ac3392/6833 [=============>................] - ETA: 0s - loss: 0.0463 - ac5056/6833 [=====================>........] - ETA: 0s - loss: 0.0449 - ac6833/6833 [==============================] - 0s 30us/step - loss: 0.0437 - acc: 0.9909
Epoch 4/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0156 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0328 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0377 - ac5184/6833 [=====================>........] - ETA: 0s - loss: 0.0413 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0425 - acc: 0.9911
Epoch 5/25
  64/6833 [..............................] - ETA: 0s - loss: 0.1212 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0465 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0407 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0412 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0415 - acc: 0.9911
Epoch 6/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0520 - ac1984/6833 [=======>......................] - ETA: 0s - loss: 0.0438 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0414 - ac5376/6833 [======================>.......] - ETA: 0s - loss: 0.0416 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0408 - acc: 0.9912
Epoch 7/25
  64/6833 [..............................] - ETA: 0s - loss: 0.1454 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0398 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0380 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0389 - ac6833/6833 [==============================] - 0s 30us/step - loss: 0.0400 - acc: 0.9912
Epoch 8/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0143 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0369 - ac3840/6833 [===============>..............] - ETA: 0s - loss: 0.0347 - ac5824/6833 [========================>.....] - ETA: 0s - loss: 0.0389 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0394 - acc: 0.9912
Epoch 9/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0667 - ac1984/6833 [=======>......................] - ETA: 0s - loss: 0.0429 - ac3840/6833 [===============>..............] - ETA: 0s - loss: 0.0381 - ac5696/6833 [========================>.....] - ETA: 0s - loss: 0.0360 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0389 - acc: 0.9914
Epoch 10/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0072 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0464 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0424 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0397 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0384 - acc: 0.9914
Epoch 11/25
  64/6833 [..............................] - ETA: 0s - loss: 0.1009 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0420 - ac3840/6833 [===============>..............] - ETA: 0s - loss: 0.0408 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0385 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0380 - acc: 0.9915
Epoch 12/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0106 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0414 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0389 - ac5312/6833 [======================>.......] - ETA: 0s - loss: 0.0383 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0376 - acc: 0.9914
Epoch 13/25
  64/6833 [..............................] - ETA: 0s - loss: 0.1149 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0410 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0348 - ac5312/6833 [======================>.......] - ETA: 0s - loss: 0.0368 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0373 - acc: 0.9918
Epoch 14/25
  64/6833 [..............................] - ETA: 0s - loss: 0.1103 - ac2048/6833 [=======>......................] - ETA: 0s - loss: 0.0387 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0381 - ac5568/6833 [=======================>......] - ETA: 0s - loss: 0.0342 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0371 - acc: 0.9917
Epoch 15/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0123 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0364 - ac3392/6833 [=============>................] - ETA: 0s - loss: 0.0319 - ac5376/6833 [======================>.......] - ETA: 0s - loss: 0.0351 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0368 - acc: 0.9915
Epoch 16/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0149 - ac2048/6833 [=======>......................] - ETA: 0s - loss: 0.0317 - ac3840/6833 [===============>..............] - ETA: 0s - loss: 0.0317 - ac5824/6833 [========================>.....] - ETA: 0s - loss: 0.0350 - ac6833/6833 [==============================] - 0s 27us/step - loss: 0.0363 - acc: 0.9915
Epoch 17/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0057 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0333 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0358 - ac5312/6833 [======================>.......] - ETA: 0s - loss: 0.0361 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0360 - acc: 0.9915
Epoch 18/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0078 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0285 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0332 - ac5248/6833 [======================>.......] - ETA: 0s - loss: 0.0336 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0357 - acc: 0.9915
Epoch 19/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0122 - ac1728/6833 [======>.......................] - ETA: 0s - loss: 0.0319 - ac3584/6833 [==============>...............] - ETA: 0s - loss: 0.0345 - ac5184/6833 [=====================>........] - ETA: 0s - loss: 0.0361 - ac6833/6833 [==============================] - 0s 30us/step - loss: 0.0355 - acc: 0.9914
Epoch 20/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0165 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0483 - ac3968/6833 [================>.............] - ETA: 0s - loss: 0.0363 - ac5824/6833 [========================>.....] - ETA: 0s - loss: 0.0339 - ac6833/6833 [==============================] - 0s 27us/step - loss: 0.0353 - acc: 0.9915
Epoch 21/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0756 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0403 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0382 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0365 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0350 - acc: 0.9915
Epoch 22/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0086 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0328 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0384 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0354 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0347 - acc: 0.9915
Epoch 23/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0186 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0355 - ac3584/6833 [==============>...............] - ETA: 0s - loss: 0.0339 - ac5376/6833 [======================>.......] - ETA: 0s - loss: 0.0342 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0345 - acc: 0.9915
Epoch 24/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0390 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0443 - ac3328/6833 [=============>................] - ETA: 0s - loss: 0.0377 - ac5120/6833 [=====================>........] - ETA: 0s - loss: 0.0381 - ac6784/6833 [============================>.] - ETA: 0s - loss: 0.0345 - ac6833/6833 [==============================] - 0s 31us/step - loss: 0.0343 - acc: 0.9915
Epoch 25/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0111 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0297 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0296 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0334 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0341 - acc: 0.9915
<keras.callbacks.History object at 0x7fbdb8bfd6d8>
>>>
>>> early_stopping_monitor = EarlyStopping(patience = 2)
>>>
>>> model.fit(X_train, y_train, epochs = 25,
...           batch_size = 64,
...           callbacks=[early_stopping_monitor])
Epoch 1/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0465 - ac1728/6833 [======>.......................] - ETA: 0s - loss: 0.0266 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0338 - ac5568/6833 [=======================>......] - ETA: 0s - loss: 0.0349 - ac6833/6833 [==============================] - 0s 30us/step - loss: 0.0338 - acc: 0.9917
Epoch 2/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0103 - ac1536/6833 [=====>........................] - ETA: 0s - loss: 0.0252 - ac3008/6833 [============>.................] - ETA: 0s - loss: 0.0334 - ac4608/6833 [===================>..........] - ETA: 0s - loss: 0.0327 - ac6272/6833 [==========================>...] - ETA: 0s - loss: 0.0320 - ac6833/6833 [==============================] - 0s 34us/step - loss: 0.0335 - acc: 0.9917
Epoch 3/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0768 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0391 - ac3520/6833 [==============>...............] - ETA: 0s - loss: 0.0385 - ac5376/6833 [======================>.......] - ETA: 0s - loss: 0.0356 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0334 - acc: 0.9918
Epoch 4/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0073 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0208 - ac3584/6833 [==============>...............] - ETA: 0s - loss: 0.0284 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0332 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0331 - acc: 0.9921
Epoch 5/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0328 - ac1984/6833 [=======>......................] - ETA: 0s - loss: 0.0259 - ac3904/6833 [================>.............] - ETA: 0s - loss: 0.0288 - ac5696/6833 [========================>.....] - ETA: 0s - loss: 0.0321 - ac6833/6833 [==============================] - 0s 27us/step - loss: 0.0329 - acc: 0.9915
Epoch 6/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0387 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0254 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0307 - ac5312/6833 [======================>.......] - ETA: 0s - loss: 0.0333 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0328 - acc: 0.9920
Epoch 7/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0501 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0508 - ac3840/6833 [===============>..............] - ETA: 0s - loss: 0.0359 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0340 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0323 - acc: 0.9918
Epoch 8/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0128 - ac1984/6833 [=======>......................] - ETA: 0s - loss: 0.0307 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0356 - ac5696/6833 [========================>.....] - ETA: 0s - loss: 0.0334 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0322 - acc: 0.9920
Epoch 9/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0121 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0293 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0282 - ac5568/6833 [=======================>......] - ETA: 0s - loss: 0.0302 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0320 - acc: 0.9917
Epoch 10/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0127 - ac1984/6833 [=======>......................] - ETA: 0s - loss: 0.0260 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0251 - ac5568/6833 [=======================>......] - ETA: 0s - loss: 0.0307 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0318 - acc: 0.9918
Epoch 11/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0093 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0444 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0357 - ac5504/6833 [=======================>......] - ETA: 0s - loss: 0.0336 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0316 - acc: 0.9917
Epoch 12/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0128 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0339 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0341 - ac5056/6833 [=====================>........] - ETA: 0s - loss: 0.0308 - ac6528/6833 [===========================>..] - ETA: 0s - loss: 0.0315 - ac6833/6833 [==============================] - 0s 31us/step - loss: 0.0314 - acc: 0.9918
Epoch 13/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0067 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0287 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0351 - ac5568/6833 [=======================>......] - ETA: 0s - loss: 0.0321 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0313 - acc: 0.9918
Epoch 14/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0046 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0337 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0338 - ac5312/6833 [======================>.......] - ETA: 0s - loss: 0.0331 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0312 - acc: 0.9918
Epoch 15/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0866 - ac1984/6833 [=======>......................] - ETA: 0s - loss: 0.0318 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0359 - ac5504/6833 [=======================>......] - ETA: 0s - loss: 0.0327 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0308 - acc: 0.9918
Epoch 16/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0411 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0292 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0298 - ac5376/6833 [======================>.......] - ETA: 0s - loss: 0.0306 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0306 - acc: 0.9918
Epoch 17/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0138 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0331 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0312 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0305 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0305 - acc: 0.9918
Epoch 18/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0070 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0249 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0265 - ac5312/6833 [======================>.......] - ETA: 0s - loss: 0.0319 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0304 - acc: 0.9918
Epoch 19/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0076 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0267 - ac3840/6833 [===============>..............] - ETA: 0s - loss: 0.0319 - ac5760/6833 [========================>.....] - ETA: 0s - loss: 0.0322 - ac6833/6833 [==============================] - 0s 27us/step - loss: 0.0301 - acc: 0.9918
Epoch 20/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0288 - ac1728/6833 [======>.......................] - ETA: 0s - loss: 0.0334 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0333 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0285 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0300 - acc: 0.9920
Epoch 21/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0076 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0357 - ac3776/6833 [===============>..............] - ETA: 0s - loss: 0.0294 - ac5696/6833 [========================>.....] - ETA: 0s - loss: 0.0279 - ac6833/6833 [==============================] - 0s 27us/step - loss: 0.0297 - acc: 0.9921
Epoch 22/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0030 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0300 - ac3904/6833 [================>.............] - ETA: 0s - loss: 0.0279 - ac5568/6833 [=======================>......] - ETA: 0s - loss: 0.0300 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0295 - acc: 0.9918
Epoch 23/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0601 - ac2048/6833 [=======>......................] - ETA: 0s - loss: 0.0279 - ac3904/6833 [================>.............] - ETA: 0s - loss: 0.0249 - ac5760/6833 [========================>.....] - ETA: 0s - loss: 0.0279 - ac6833/6833 [==============================] - 0s 28us/step - loss: 0.0294 - acc: 0.9920
Epoch 24/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0828 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0305 - ac3904/6833 [================>.............] - ETA: 0s - loss: 0.0296 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0312 - ac6833/6833 [==============================] - 0s 27us/step - loss: 0.0291 - acc: 0.9922
Epoch 25/25
  64/6833 [..............................] - ETA: 0s - loss: 0.0154 - ac1920/6833 [=======>......................] - ETA: 0s - loss: 0.0152 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0231 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0281 - ac6833/6833 [==============================] - 0s 29us/step - loss: 0.0289 - acc: 0.9920
<keras.callbacks.History object at 0x7fbdb8bfd2b0>
>>>
>>>
>>> early_stopping_monitor = EarlyStopping(patience = 2)
>>>
>>> loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
2278/2278 [==============================] - 0s 18us/step
>>> y_hat_nnet = model.predict(x_test, batch_size=128)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x_test' is not defined
>>> y_hat_nnet = model.predict(X_test, batch_size=128)
>>> y_hat_nnet.sum()
24.097439
>>> y_hat
array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
>>> print y_hat_nnet
  File "<stdin>", line 1
    print y_hat_nnet
                   ^
SyntaxError: Missing parentheses in call to 'print'. Did you mean print(y_hat_nnet)?
>>> print(y_hat_nnet)
[[  1.57060381e-03]
 [  3.38791619e-06]
 [  2.87447067e-04]
 ...,
 [  2.93153082e-03]
 [  9.12566393e-05]
 [  3.33495531e-03]]
>>> to_categorical(y_train)
array([[ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.],
       ...,
       [ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.]])
>>> to_categorical(y_train).sum()
6833.0
>>> y_train.sum()
67.0
>>> len(y_train)
6833
>>>
>>> # 1. Specify
... model = Sequential()
>>> n_cols = X_train.shape[1]
>>> shape = (n_cols,)
>>> np.round(n_cols * 2/3, 0) # nodes in layer 1
17.0
>>> # nodes in layer 2 = half of those in layer 1
... model.add(Dense(17, activation='relu', input_shape=shape))
>>> #model.add(Dense(8, activation='relu', input_shape=shape))
... model.add(Dense(2, activation='softmax'))
>>> print(model.summary())
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_3 (Dense)              (None, 17)                442
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 36
=================================================================
Total params: 478
Trainable params: 478
Non-trainable params: 0
_________________________________________________________________
None
>>>
>>> # 2. Compile
... model.compile(optimizer='adam',
...               loss='binary_crossentropy',
...               metrics=['accuracy'])
>>>
>>> # 3. Fit :
... model.fit(X_train, to_categorical(y_train), epochs = 5)
Epoch 1/5
  32/6833 [..............................] - ETA: 43s - loss: 0.6265 - a 864/6833 [==>...........................] - ETA: 1s - loss: 0.6184 - ac1824/6833 [=======>......................] - ETA: 0s - loss: 0.5021 - ac2688/6833 [==========>...................] - ETA: 0s - loss: 0.4235 - ac3552/6833 [==============>...............] - ETA: 0s - loss: 0.3644 - ac4576/6833 [===================>..........] - ETA: 0s - loss: 0.3159 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.2894 - ac6304/6833 [==========================>...] - ETA: 0s - loss: 0.2639 - ac6833/6833 [==============================] - 1s 87us/step - loss: 0.2502 - acc: 0.9341
Epoch 2/5
  32/6833 [..............................] - ETA: 0s - loss: 0.0523 - ac 992/6833 [===>..........................] - ETA: 0s - loss: 0.0925 - ac1888/6833 [=======>......................] - ETA: 0s - loss: 0.0803 - ac2816/6833 [===========>..................] - ETA: 0s - loss: 0.0740 - ac3584/6833 [==============>...............] - ETA: 0s - loss: 0.0794 - ac4480/6833 [==================>...........] - ETA: 0s - loss: 0.0767 - ac5344/6833 [======================>.......] - ETA: 0s - loss: 0.0731 - ac6304/6833 [==========================>...] - ETA: 0s - loss: 0.0711 - ac6833/6833 [==============================] - 0s 57us/step - loss: 0.0690 - acc: 0.9903
Epoch 3/5
  32/6833 [..............................] - ETA: 0s - loss: 0.1871 - ac 928/6833 [===>..........................] - ETA: 0s - loss: 0.0796 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0666 - ac2560/6833 [==========>...................] - ETA: 0s - loss: 0.0620 - ac3520/6833 [==============>...............] - ETA: 0s - loss: 0.0612 - ac4416/6833 [==================>...........] - ETA: 0s - loss: 0.0544 - ac5344/6833 [======================>.......] - ETA: 0s - loss: 0.0515 - ac6304/6833 [==========================>...] - ETA: 0s - loss: 0.0514 - ac6833/6833 [==============================] - 0s 57us/step - loss: 0.0517 - acc: 0.9903
Epoch 4/5
  32/6833 [..............................] - ETA: 0s - loss: 0.0533 - ac 960/6833 [===>..........................] - ETA: 0s - loss: 0.0416 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0382 - ac2720/6833 [==========>...................] - ETA: 0s - loss: 0.0407 - ac3584/6833 [==============>...............] - ETA: 0s - loss: 0.0428 - ac4416/6833 [==================>...........] - ETA: 0s - loss: 0.0465 - ac5216/6833 [=====================>........] - ETA: 0s - loss: 0.0462 - ac6048/6833 [=========================>....] - ETA: 0s - loss: 0.0448 - ac6833/6833 [==============================] - 0s 59us/step - loss: 0.0454 - acc: 0.9903
Epoch 5/5
  32/6833 [..............................] - ETA: 0s - loss: 0.1603 - ac 896/6833 [==>...........................] - ETA: 0s - loss: 0.0482 - ac1696/6833 [======>.......................] - ETA: 0s - loss: 0.0408 - ac2560/6833 [==========>...................] - ETA: 0s - loss: 0.0430 - ac3328/6833 [=============>................] - ETA: 0s - loss: 0.0469 - ac4256/6833 [=================>............] - ETA: 0s - loss: 0.0463 - ac5120/6833 [=====================>........] - ETA: 0s - loss: 0.0477 - ac5984/6833 [=========================>....] - ETA: 0s - loss: 0.0444 - ac6688/6833 [============================>.] - ETA: 0s - loss: 0.0429 - ac6833/6833 [==============================] - 0s 62us/step - loss: 0.0422 - acc: 0.9909
<keras.callbacks.History object at 0x7fbd9c687fd0>
>>>
>>> # 5. Predict
... loss_and_metrics = model.evaluate(X_test, to_categorical(y_test), batch_size=128)
2278/2278 [==============================] - 0s 21us/step
>>> y_hat_nnet = model.predict(X_test, batch_size=128)
>>> y_hat_nnet.sum()
2278.0
>>> y_hat_nnet
array([[  9.99507546e-01,   4.92438558e-04],
       [  9.99883413e-01,   1.16524105e-04],
       [  9.99659300e-01,   3.40718951e-04],
       ...,
       [  9.98847008e-01,   1.15302065e-03],
       [  9.95165229e-01,   4.83480329e-03],
       [  9.89332795e-01,   1.06672812e-02]], dtype=float32)
>>> np.round(y_hat_nnet)
array([[ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.],
       ...,
       [ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.]], dtype=float32)
>>> np.round(y_hat_nnet, 2)
array([[ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       ...,
       [ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       [ 0.99000001,  0.01      ]], dtype=float32)
>>> np.round(y_hat_nnet, 3)
array([[ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       ...,
       [ 0.99900001,  0.001     ],
       [ 0.995     ,  0.005     ],
       [ 0.98900002,  0.011     ]], dtype=float32)
>>> np.round(y_hat_nnet, 4)
array([[  9.99499977e-01,   5.00000024e-04],
       [  9.99899983e-01,   9.99999975e-05],
       [  9.99700010e-01,   3.00000014e-04],
       ...,
       [  9.98799980e-01,   1.20000006e-03],
       [  9.95199978e-01,   4.80000023e-03],
       [  9.89300013e-01,   1.07000005e-02]], dtype=float32)
>>> np.round(y_hat_nnet, 3)
array([[ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       ...,
       [ 0.99900001,  0.001     ],
       [ 0.995     ,  0.005     ],
       [ 0.98900002,  0.011     ]], dtype=float32)
>>>
>>> y_hat_nnet_pred = model.predict(X_test, batch_size=128)
>>> y_hat_nnet_pred.sum()
2278.0
>>> np.round(y_hat_nnet, 3) #here we see it can be [0.995, 0.005]
array([[ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       ...,
       [ 0.99900001,  0.001     ],
       [ 0.995     ,  0.005     ],
       [ 0.98900002,  0.011     ]], dtype=float32)
>>> # final predictions are the rounded version:
... y_hat_nnet = np.round(y_hat_nnet_pred, 1)
>>> y_hat_nnet
array([[ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.],
       ...,
       [ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.]], dtype=float32)
>>> y_hat_nnet_pred.sum()
2278.0
>>>
>>>
>>>
>>> model = Sequential()
>>> n_cols = X_train.shape[1]
>>> shape = (n_cols,)
>>> np.round(n_cols * 2/3, 0) # nodes in layer 1
17.0
>>> # nodes in layer 2 = half of those in layer 1
... model.add(Dense(17, activation='relu', input_shape=shape))
>>> #model.add(Dense(8, activation='relu', input_shape=shape))
... model.add(Dense(1, activation='sigmoid'))
>>> #model.add(Dense(2, activation='softmax'))
... print(model.summary())
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_5 (Dense)              (None, 17)                442
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 18
=================================================================
Total params: 460
Trainable params: 460
Non-trainable params: 0
_________________________________________________________________
None
>>>
>>> # 2. Compile
... model.compile(optimizer='adam',
...               loss='binary_crossentropy',
...               metrics=['accuracy'])
>>>
>>> # 3. Fit :
... model.fit(X_train, y_train, epochs = 5)
Epoch 1/5
  32/6833 [..............................] - ETA: 51s - loss: 0.4941 - a 800/6833 [==>...........................] - ETA: 2s - loss: 0.3880 - ac1632/6833 [======>.......................] - ETA: 1s - loss: 0.3461 - ac2528/6833 [==========>...................] - ETA: 0s - loss: 0.3059 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.2752 - ac4352/6833 [==================>...........] - ETA: 0s - loss: 0.2491 - ac5344/6833 [======================>.......] - ETA: 0s - loss: 0.2260 - ac6272/6833 [==========================>...] - ETA: 0s - loss: 0.2048 - ac6833/6833 [==============================] - 1s 92us/step - loss: 0.1983 - acc: 0.9814
Epoch 2/5
  32/6833 [..............................] - ETA: 0s - loss: 0.0566 - ac 992/6833 [===>..........................] - ETA: 0s - loss: 0.0890 - ac1856/6833 [=======>......................] - ETA: 0s - loss: 0.0875 - ac2816/6833 [===========>..................] - ETA: 0s - loss: 0.0837 - ac3552/6833 [==============>...............] - ETA: 0s - loss: 0.0821 - ac4480/6833 [==================>...........] - ETA: 0s - loss: 0.0779 - ac5440/6833 [======================>.......] - ETA: 0s - loss: 0.0749 - ac6432/6833 [===========================>..] - ETA: 0s - loss: 0.0725 - ac6833/6833 [==============================] - 0s 55us/step - loss: 0.0697 - acc: 0.9903
Epoch 3/5
  32/6833 [..............................] - ETA: 0s - loss: 0.0241 - ac 960/6833 [===>..........................] - ETA: 0s - loss: 0.0557 - ac1824/6833 [=======>......................] - ETA: 0s - loss: 0.0574 - ac2784/6833 [===========>..................] - ETA: 0s - loss: 0.0589 - ac3488/6833 [==============>...............] - ETA: 0s - loss: 0.0537 - ac4448/6833 [==================>...........] - ETA: 0s - loss: 0.0550 - ac5376/6833 [======================>.......] - ETA: 0s - loss: 0.0540 - ac6336/6833 [==========================>...] - ETA: 0s - loss: 0.0536 - ac6833/6833 [==============================] - 0s 57us/step - loss: 0.0543 - acc: 0.9902
Epoch 4/5
  32/6833 [..............................] - ETA: 0s - loss: 0.1502 - ac 896/6833 [==>...........................] - ETA: 0s - loss: 0.0400 - ac1824/6833 [=======>......................] - ETA: 0s - loss: 0.0353 - ac2624/6833 [==========>...................] - ETA: 0s - loss: 0.0418 - ac3488/6833 [==============>...............] - ETA: 0s - loss: 0.0449 - ac4480/6833 [==================>...........] - ETA: 0s - loss: 0.0483 - ac5504/6833 [=======================>......] - ETA: 0s - loss: 0.0504 - ac6400/6833 [===========================>..] - ETA: 0s - loss: 0.0486 - ac6833/6833 [==============================] - 0s 56us/step - loss: 0.0480 - acc: 0.9902
Epoch 5/5
  32/6833 [..............................] - ETA: 0s - loss: 0.0075 - ac 960/6833 [===>..........................] - ETA: 0s - loss: 0.0349 - ac1824/6833 [=======>......................] - ETA: 0s - loss: 0.0429 - ac2720/6833 [==========>...................] - ETA: 0s - loss: 0.0508 - ac3648/6833 [===============>..............] - ETA: 0s - loss: 0.0457 - ac4608/6833 [===================>..........] - ETA: 0s - loss: 0.0473 - ac5472/6833 [=======================>......] - ETA: 0s - loss: 0.0455 - ac6432/6833 [===========================>..] - ETA: 0s - loss: 0.0456 - ac6833/6833 [==============================] - 0s 56us/step - loss: 0.0446 - acc: 0.9902
<keras.callbacks.History object at 0x7fbd9c6205f8>
>>> # Accuracy = (TP + TN) / (P+N).    T: True. P: positives. N: negatives.
... # Accuracy = 0.99 for this model
...
>>> # Fit using early stopping
... early_stopping_monitor = EarlyStopping(patience = 2)
>>>
>>> model.fit(X_train, y_train, epochs = 25,
...           batch_size = 128,
...           callbacks=[early_stopping_monitor])
Epoch 1/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0355 - ac3328/6833 [=============>................] - ETA: 0s - loss: 0.0407 - ac6400/6833 [===========================>..] - ETA: 0s - loss: 0.0419 - ac6833/6833 [==============================] - 0s 17us/step - loss: 0.0424 - acc: 0.9905
Epoch 2/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0616 - ac2816/6833 [===========>..................] - ETA: 0s - loss: 0.0388 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0395 - ac6833/6833 [==============================] - 0s 19us/step - loss: 0.0420 - acc: 0.9906
Epoch 3/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0153 - ac3328/6833 [=============>................] - ETA: 0s - loss: 0.0464 - ac6144/6833 [=========================>....] - ETA: 0s - loss: 0.0412 - ac6833/6833 [==============================] - 0s 18us/step - loss: 0.0415 - acc: 0.9906
Epoch 4/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0253 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0403 - ac6400/6833 [===========================>..] - ETA: 0s - loss: 0.0414 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0411 - acc: 0.9908
Epoch 5/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0315 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0386 - ac6656/6833 [============================>.] - ETA: 0s - loss: 0.0415 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0407 - acc: 0.9908
Epoch 6/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0397 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0385 - ac6528/6833 [===========================>..] - ETA: 0s - loss: 0.0409 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0403 - acc: 0.9908
Epoch 7/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0560 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0397 - ac6833/6833 [==============================] - 0s 15us/step - loss: 0.0400 - acc: 0.9909
Epoch 8/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0252 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0343 - ac6833/6833 [==============================] - 0s 15us/step - loss: 0.0396 - acc: 0.9906
Epoch 9/25
 128/6833 [..............................] - ETA: 0s - loss: 0.1013 - ac3328/6833 [=============>................] - ETA: 0s - loss: 0.0389 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0402 - ac6833/6833 [==============================] - 0s 20us/step - loss: 0.0393 - acc: 0.9909
Epoch 10/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0122 - ac2304/6833 [=========>....................] - ETA: 0s - loss: 0.0331 - ac3968/6833 [================>.............] - ETA: 0s - loss: 0.0376 - ac5888/6833 [========================>.....] - ETA: 0s - loss: 0.0381 - ac6833/6833 [==============================] - 0s 30us/step - loss: 0.0391 - acc: 0.9909
Epoch 11/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0440 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0334 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0448 - ac5760/6833 [========================>.....] - ETA: 0s - loss: 0.0406 - ac6833/6833 [==============================] - 0s 27us/step - loss: 0.0388 - acc: 0.9909
Epoch 12/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0384 - ac2688/6833 [==========>...................] - ETA: 0s - loss: 0.0382 - ac6016/6833 [=========================>....] - ETA: 0s - loss: 0.0399 - ac6833/6833 [==============================] - 0s 18us/step - loss: 0.0384 - acc: 0.9912
Epoch 13/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0065 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0437 - ac5760/6833 [========================>.....] - ETA: 0s - loss: 0.0394 - ac6833/6833 [==============================] - 0s 18us/step - loss: 0.0382 - acc: 0.9912
Epoch 14/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0153 - ac2944/6833 [===========>..................] - ETA: 0s - loss: 0.0385 - ac6528/6833 [===========================>..] - ETA: 0s - loss: 0.0364 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0379 - acc: 0.9911
Epoch 15/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0543 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0334 - ac6400/6833 [===========================>..] - ETA: 0s - loss: 0.0364 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0376 - acc: 0.9912
Epoch 16/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0291 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0315 - ac6784/6833 [============================>.] - ETA: 0s - loss: 0.0372 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0374 - acc: 0.9912
Epoch 17/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0158 - ac2816/6833 [===========>..................] - ETA: 0s - loss: 0.0344 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0358 - ac6833/6833 [==============================] - 0s 19us/step - loss: 0.0371 - acc: 0.9912
Epoch 18/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0191 - ac2688/6833 [==========>...................] - ETA: 0s - loss: 0.0347 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0396 - ac6833/6833 [==============================] - 0s 19us/step - loss: 0.0369 - acc: 0.9912
Epoch 19/25
 128/6833 [..............................] - ETA: 0s - loss: 0.1153 - ac2816/6833 [===========>..................] - ETA: 0s - loss: 0.0325 - ac5760/6833 [========================>.....] - ETA: 0s - loss: 0.0346 - ac6833/6833 [==============================] - 0s 19us/step - loss: 0.0367 - acc: 0.9914
Epoch 20/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0263 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0322 - ac6144/6833 [=========================>....] - ETA: 0s - loss: 0.0361 - ac6833/6833 [==============================] - 0s 18us/step - loss: 0.0364 - acc: 0.9914
Epoch 21/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0414 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0360 - ac6400/6833 [===========================>..] - ETA: 0s - loss: 0.0352 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0362 - acc: 0.9912
Epoch 22/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0360 - ac3200/6833 [=============>................] - ETA: 0s - loss: 0.0355 - ac6016/6833 [=========================>....] - ETA: 0s - loss: 0.0351 - ac6833/6833 [==============================] - 0s 18us/step - loss: 0.0359 - acc: 0.9914
Epoch 23/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0655 - ac3712/6833 [===============>..............] - ETA: 0s - loss: 0.0319 - ac6833/6833 [==============================] - 0s 14us/step - loss: 0.0357 - acc: 0.9915
Epoch 24/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0146 - ac3072/6833 [============>.................] - ETA: 0s - loss: 0.0290 - ac6656/6833 [============================>.] - ETA: 0s - loss: 0.0359 - ac6833/6833 [==============================] - 0s 16us/step - loss: 0.0355 - acc: 0.9914
Epoch 25/25
 128/6833 [..............................] - ETA: 0s - loss: 0.0372 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0336 - ac6833/6833 [==============================] - 0s 15us/step - loss: 0.0353 - acc: 0.9914
<keras.callbacks.History object at 0x7fbdb8bfd6d8>
>>>
>>>
>>> # 5. Predict
... loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
2278/2278 [==============================] - 0s 26us/step
>>> y_hat_nnet = model.predict(X_test, batch_size=128)
>>> y_hat_nnet.sum()
23.726162
>>> # qq den ska bara kunna predicta 0 och 1 ?
...
>>> y_hat_nnet.sum()
23.726162
>>> y_hat_nnet
array([[  2.10596321e-04],
       [  1.10051815e-05],
       [  1.23276739e-04],
       ...,
       [  3.04891332e-03],
       [  2.19623529e-04],
       [  1.00993542e-02]], dtype=float32)
>>> np.round(y_hat_nnet, 2)
array([[ 0.  ],
       [ 0.  ],
       [ 0.  ],
       ...,
       [ 0.  ],
       [ 0.  ],
       [ 0.01]], dtype=float32)
>>> np.round(y_hat_nnet, 3)
array([[ 0.   ],
       [ 0.   ],
       [ 0.   ],
       ...,
       [ 0.003],
       [ 0.   ],
       [ 0.01 ]], dtype=float32)
>>> np.round(y_hat_nnet, 4)
array([[  1.99999995e-04],
       [  0.00000000e+00],
       [  9.99999975e-05],
       ...,
       [  3.00000003e-03],
       [  1.99999995e-04],
       [  1.00999996e-02]], dtype=float32)
>>> np.round(y_hat_nnet, 3)
array([[ 0.   ],
       [ 0.   ],
       [ 0.   ],
       ...,
       [ 0.003],
       [ 0.   ],
       [ 0.01 ]], dtype=float32)
>>> np.round(y_hat_nnet, 3)[1:100]
array([[ 0.        ],
       [ 0.        ],
       [ 0.001     ],
       [ 0.003     ],
       [ 0.001     ],
       [ 0.001     ],
       [ 0.017     ],
       [ 0.        ],
       [ 0.        ],
       [ 0.001     ],
       [ 0.001     ],
       [ 0.001     ],
       [ 0.002     ],
       [ 0.062     ],
       [ 0.        ],
       [ 0.003     ],
       [ 0.008     ],
       [ 0.026     ],
       [ 0.        ],
       [ 0.        ],
       [ 0.002     ],
       [ 0.        ],
       [ 0.        ],
       [ 0.004     ],
       [ 0.003     ],
       [ 0.002     ],
       [ 0.001     ],
       [ 0.002     ],
       [ 0.001     ],
       [ 0.        ],
       [ 0.011     ],
       [ 0.        ],
       [ 0.002     ],
       [ 0.        ],
       [ 0.001     ],
       [ 0.043     ],
       [ 0.001     ],
       [ 0.005     ],
       [ 0.001     ],
       [ 0.011     ],
       [ 0.104     ],
       [ 0.008     ],
       [ 0.002     ],
       [ 0.005     ],
       [ 0.        ],
       [ 0.001     ],
       [ 0.15800001],
       [ 0.011     ],
       [ 0.002     ],
       [ 0.011     ],
       [ 0.011     ],
       [ 0.001     ],
       [ 0.079     ],
       [ 0.        ],
       [ 0.001     ],
       [ 0.138     ],
       [ 0.001     ],
       [ 0.002     ],
       [ 0.        ],
       [ 0.012     ],
       [ 0.002     ],
       [ 0.003     ],
       [ 0.        ],
       [ 0.003     ],
       [ 0.074     ],
       [ 0.008     ],
       [ 0.001     ],
       [ 0.        ],
       [ 0.008     ],
       [ 0.107     ],
       [ 0.042     ],
       [ 0.004     ],
       [ 0.        ],
       [ 0.028     ],
       [ 0.        ],
       [ 0.001     ],
       [ 0.        ],
       [ 0.004     ],
       [ 0.002     ],
       [ 0.        ],
       [ 0.        ],
       [ 0.001     ],
       [ 0.002     ],
       [ 0.003     ],
       [ 0.019     ],
       [ 0.001     ],
       [ 0.001     ],
       [ 0.087     ],
       [ 0.035     ],
       [ 0.002     ],
       [ 0.001     ],
       [ 0.        ],
       [ 0.01      ],
       [ 0.        ],
       [ 0.039     ],
       [ 0.001     ],
       [ 0.        ],
       [ 0.        ],
       [ 0.        ]], dtype=float32)
>>>
>>> (y_hat_nnet > 0.5).sum()
2
>>> (y_hat_nnet > 0.4).sum()
3
>>> (y_hat_nnet > 0.3).sum()
11
>>> (y_hat_nnet > 0.2).sum()
23
>>> (y_hat_nnet > 0.7).sum()
0
>>>
>>> # 1. Specify
... model = Sequential()
>>> n_cols = X_train.shape[1]
>>> shape = (n_cols,)
>>> np.round(n_cols * 2/3, 0) # nodes in layer 1
17.0
>>> # nodes in layer 2 = half of those in layer 1
... model.add(Dense(17, activation='relu', input_shape=shape))
>>> model.add(Dense(8, activation='relu', input_shape=shape))
>>> model.add(Dense(2, activation='softmax'))
>>> print(model.summary())
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_7 (Dense)              (None, 17)                442
_________________________________________________________________
dense_8 (Dense)              (None, 8)                 144
_________________________________________________________________
dense_9 (Dense)              (None, 2)                 18
=================================================================
Total params: 604
Trainable params: 604
Non-trainable params: 0
_________________________________________________________________
None
>>>
>>> # 2. Compile
... model.compile(optimizer='adam',
...               loss='binary_crossentropy',
...               metrics=['accuracy'])
>>>
>>> # 3. Fit :
... model.fit(X_train, to_categorical(y_train), epochs = 10)
Epoch 1/10
  32/6833 [..............................] - ETA: 9:43 - loss: 0.5052 -   96/6833 [..............................] - ETA: 3:16 - loss: 0.4484 -  320/6833 [>.............................] - ETA: 58s - loss: 0.4276 - a 992/6833 [===>..........................] - ETA: 17s - loss: 0.3542 - a1792/6833 [======>.......................] - ETA: 8s - loss: 0.2989 - ac2592/6833 [==========>...................] - ETA: 4s - loss: 0.2496 - ac3488/6833 [==============>...............] - ETA: 2s - loss: 0.2145 - ac4416/6833 [==================>...........] - ETA: 1s - loss: 0.1838 - ac5344/6833 [======================>.......] - ETA: 0s - loss: 0.1724 - ac6048/6833 [=========================>....] - ETA: 0s - loss: 0.1589 - ac6833/6833 [==============================] - 3s 477us/step - loss: 0.1479 - acc: 0.9829
Epoch 2/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0219 - ac 800/6833 [==>...........................] - ETA: 0s - loss: 0.1016 - ac1696/6833 [======>.......................] - ETA: 0s - loss: 0.0794 - ac2560/6833 [==========>...................] - ETA: 0s - loss: 0.0705 - ac3392/6833 [=============>................] - ETA: 0s - loss: 0.0595 - ac4288/6833 [=================>............] - ETA: 0s - loss: 0.0603 - ac5216/6833 [=====================>........] - ETA: 0s - loss: 0.0592 - ac6048/6833 [=========================>....] - ETA: 0s - loss: 0.0608 - ac6833/6833 [==============================] - 0s 58us/step - loss: 0.0605 - acc: 0.9902
Epoch 3/10
  32/6833 [..............................] - ETA: 0s - loss: 0.1976 - ac 832/6833 [==>...........................] - ETA: 0s - loss: 0.0534 - ac1760/6833 [======>.......................] - ETA: 0s - loss: 0.0483 - ac2720/6833 [==========>...................] - ETA: 0s - loss: 0.0502 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0538 - ac4448/6833 [==================>...........] - ETA: 0s - loss: 0.0519 - ac5248/6833 [======================>.......] - ETA: 0s - loss: 0.0499 - ac5952/6833 [=========================>....] - ETA: 0s - loss: 0.0481 - ac6688/6833 [============================>.] - ETA: 0s - loss: 0.0479 - ac6833/6833 [==============================] - 0s 62us/step - loss: 0.0474 - acc: 0.9903
Epoch 4/10
  32/6833 [..............................] - ETA: 0s - loss: 0.1989 - ac 800/6833 [==>...........................] - ETA: 0s - loss: 0.0452 - ac1536/6833 [=====>........................] - ETA: 0s - loss: 0.0374 - ac2464/6833 [=========>....................] - ETA: 0s - loss: 0.0354 - ac3296/6833 [=============>................] - ETA: 0s - loss: 0.0425 - ac4032/6833 [================>.............] - ETA: 0s - loss: 0.0405 - ac4896/6833 [====================>.........] - ETA: 0s - loss: 0.0393 - ac5792/6833 [========================>.....] - ETA: 0s - loss: 0.0422 - ac6720/6833 [============================>.] - ETA: 0s - loss: 0.0401 - ac6833/6833 [==============================] - 0s 61us/step - loss: 0.0411 - acc: 0.9908
Epoch 5/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0630 - ac 768/6833 [==>...........................] - ETA: 0s - loss: 0.0300 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0383 - ac2496/6833 [=========>....................] - ETA: 0s - loss: 0.0497 - ac3296/6833 [=============>................] - ETA: 0s - loss: 0.0426 - ac4192/6833 [=================>............] - ETA: 0s - loss: 0.0387 - ac5120/6833 [=====================>........] - ETA: 0s - loss: 0.0375 - ac5952/6833 [=========================>....] - ETA: 0s - loss: 0.0363 - ac6816/6833 [============================>.] - ETA: 0s - loss: 0.0367 - ac6833/6833 [==============================] - 0s 60us/step - loss: 0.0373 - acc: 0.9909
Epoch 6/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0043 - ac 768/6833 [==>...........................] - ETA: 0s - loss: 0.0306 - ac1536/6833 [=====>........................] - ETA: 0s - loss: 0.0346 - ac2208/6833 [========>.....................] - ETA: 0s - loss: 0.0358 - ac2944/6833 [===========>..................] - ETA: 0s - loss: 0.0346 - ac3584/6833 [==============>...............] - ETA: 0s - loss: 0.0385 - ac4416/6833 [==================>...........] - ETA: 0s - loss: 0.0374 - ac5216/6833 [=====================>........] - ETA: 0s - loss: 0.0362 - ac6016/6833 [=========================>....] - ETA: 0s - loss: 0.0353 - ac6833/6833 [==============================] - 0s 67us/step - loss: 0.0354 - acc: 0.9912
Epoch 7/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0160 - ac 928/6833 [===>..........................] - ETA: 0s - loss: 0.0388 - ac1728/6833 [======>.......................] - ETA: 0s - loss: 0.0375 - ac2560/6833 [==========>...................] - ETA: 0s - loss: 0.0353 - ac3392/6833 [=============>................] - ETA: 0s - loss: 0.0390 - ac4256/6833 [=================>............] - ETA: 0s - loss: 0.0373 - ac5088/6833 [=====================>........] - ETA: 0s - loss: 0.0350 - ac5984/6833 [=========================>....] - ETA: 0s - loss: 0.0350 - ac6752/6833 [============================>.] - ETA: 0s - loss: 0.0341 - ac6833/6833 [==============================] - 0s 60us/step - loss: 0.0343 - acc: 0.9912
Epoch 8/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0054 - ac 864/6833 [==>...........................] - ETA: 0s - loss: 0.0164 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0203 - ac2560/6833 [==========>...................] - ETA: 0s - loss: 0.0208 - ac3424/6833 [==============>...............] - ETA: 0s - loss: 0.0270 - ac4160/6833 [=================>............] - ETA: 0s - loss: 0.0294 - ac4992/6833 [====================>.........] - ETA: 0s - loss: 0.0307 - ac5632/6833 [=======================>......] - ETA: 0s - loss: 0.0305 - ac6400/6833 [===========================>..] - ETA: 0s - loss: 0.0314 - ac6833/6833 [==============================] - 0s 65us/step - loss: 0.0329 - acc: 0.9909
Epoch 9/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0026 - ac 768/6833 [==>...........................] - ETA: 0s - loss: 0.0275 - ac1568/6833 [=====>........................] - ETA: 0s - loss: 0.0255 - ac2368/6833 [=========>....................] - ETA: 0s - loss: 0.0258 - ac3264/6833 [=============>................] - ETA: 0s - loss: 0.0249 - ac4064/6833 [================>.............] - ETA: 0s - loss: 0.0236 - ac4960/6833 [====================>.........] - ETA: 0s - loss: 0.0262 - ac5728/6833 [========================>.....] - ETA: 0s - loss: 0.0313 - ac6464/6833 [===========================>..] - ETA: 0s - loss: 0.0319 - ac6833/6833 [==============================] - 0s 65us/step - loss: 0.0323 - acc: 0.9914
Epoch 10/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0094 - ac 864/6833 [==>...........................] - ETA: 0s - loss: 0.0303 - ac1696/6833 [======>.......................] - ETA: 0s - loss: 0.0362 - ac2464/6833 [=========>....................] - ETA: 0s - loss: 0.0361 - ac3168/6833 [============>.................] - ETA: 0s - loss: 0.0354 - ac3808/6833 [===============>..............] - ETA: 0s - loss: 0.0337 - ac4608/6833 [===================>..........] - ETA: 0s - loss: 0.0320 - ac5280/6833 [======================>.......] - ETA: 0s - loss: 0.0305 - ac6048/6833 [=========================>....] - ETA: 0s - loss: 0.0313 - ac6816/6833 [============================>.] - ETA: 0s - loss: 0.0317 - ac6833/6833 [==============================] - 0s 68us/step - loss: 0.0316 - acc: 0.9911
<keras.callbacks.History object at 0x7fbd96045198>
>>>
>>> # 5. Predict
... loss_and_metrics = model.evaluate(X_test, to_categorical(y_test))
2278/2278 [==============================] - 0s 99us/step
>>> y_hat_nnet_pred = model.predict(X_test)
>>> y_hat_nnet_pred.sum()
2278.0
>>> np.round(y_hat_nnet, 3) #here we see it can be [0.995, 0.005]
array([[ 0.   ],
       [ 0.   ],
       [ 0.   ],
       ...,
       [ 0.003],
       [ 0.   ],
       [ 0.01 ]], dtype=float32)
>>> dfm.colnames
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/pandas/core/generic.py", line 3081, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'colnames'
>>> dfm.columns
Index(['uuid', 'default', 'account_amount_added_12_24m',
       'account_days_in_dc_12_24m', 'account_days_in_rem_12_24m',
       'account_days_in_term_12_24m', 'account_incoming_debt_vs_paid_0_24m',
       'account_status', 'account_worst_status_0_3m',
       'account_worst_status_12_24m', 'account_worst_status_3_6m',
       'account_worst_status_6_12m', 'age', 'avg_payment_span_0_12m',
       'avg_payment_span_0_3m', 'merchant_category', 'merchant_group',
       'has_paid', 'max_paid_inv_0_12m', 'max_paid_inv_0_24m', 'name_in_email',
       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
       'num_arch_ok_12_24m', 'num_arch_rem_0_12m',
       'num_arch_written_off_0_12m', 'num_arch_written_off_12_24m',
       'num_unpaid_bills', 'status_last_archived_0_24m',
       'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m',
       'status_max_archived_0_6_months', 'status_max_archived_0_12_months',
       'status_max_archived_0_24_months', 'recovery_debt',
       'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
       'sum_paid_inv_0_12m', 'time_hours', 'worst_status_active_inv'],
      dtype='object')
>>> vardescr
                               Variable         Type
0                                  uuid         text
1                               default  categorical
2           account_amount_added_12_24m      numeric
3             account_days_in_dc_12_24m      numeric
4            account_days_in_rem_12_24m      numeric
5           account_days_in_term_12_24m      numeric
6   account_incoming_debt_vs_paid_0_24m      numeric
7                        account_status  categorical
8             account_worst_status_0_3m  categorical
9           account_worst_status_12_24m  categorical
10            account_worst_status_3_6m  categorical
11           account_worst_status_6_12m  categorical
12                                  age      numeric
13               avg_payment_span_0_12m      numeric
14                avg_payment_span_0_3m      numeric
15                    merchant_category  categorical
16                       merchant_group  categorical
17                             has_paid      boolean
18                   max_paid_inv_0_12m      numeric
19                   max_paid_inv_0_24m      numeric
20                        name_in_email  categorical
21     num_active_div_by_paid_inv_0_12m      numeric
22                       num_active_inv      numeric
23                    num_arch_dc_0_12m      numeric
24                   num_arch_dc_12_24m      numeric
25                    num_arch_ok_0_12m      numeric
26                   num_arch_ok_12_24m      numeric
27                   num_arch_rem_0_12m      numeric
28           num_arch_written_off_0_12m      numeric
29          num_arch_written_off_12_24m      numeric
30                     num_unpaid_bills      numeric
31           status_last_archived_0_24m  categorical
32       status_2nd_last_archived_0_24m  categorical
33       status_3rd_last_archived_0_24m  categorical
34       status_max_archived_0_6_months  categorical
35      status_max_archived_0_12_months  categorical
36      status_max_archived_0_24_months  categorical
37                        recovery_debt      numeric
38       sum_capital_paid_account_0_12m      numeric
39      sum_capital_paid_account_12_24m      numeric
40                   sum_paid_inv_0_12m      numeric
41                           time_hours      numeric
42              worst_status_active_inv  categorical
>>> x1 = dfm['account_status']
>>> encoder = LabelEncoder()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'LabelEncoder' is not defined
>>> from sklearn.preprocessing import LabelEncoder
>>> x1 = dfm['account_status']
>>> encoder = LabelEncoder()
>>> encoder.fit(x1)
LabelEncoder()
>>> encoded_x1 = encoder.transform(x1)
>>> encoded_x1
array([0, 0, 0, ..., 0, 0, 0])
>>> encoded_x1[1:100]
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 0, 0, 0, 0])
>>> encoded_x1.shape
(9111,)
>>> n_cols * 2/3 * 0.5
8.333333333333334
>>> np.round(n_cols * 2/3, 0) # nodes in layer 1
17.0
>>>
>>> model = Sequential()
>>> n_cols = X_train.shape[1]
>>> shape = (n_cols,)
>>> n_cols * 2/3       # nodes in layer 1
16.666666666666668
>>> n_cols * 2/3 * 0.5 # nodes in layer 2
8.333333333333334
>>> model.add(Dense(17, activation='relu', input_shape=shape))
>>> model.add(Dense(8, activation='relu', input_shape=shape))
>>> model.add(Dense(2, activation='softmax'))
>>> print(model.summary())
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_10 (Dense)             (None, 17)                442
_________________________________________________________________
dense_11 (Dense)             (None, 8)                 144
_________________________________________________________________
dense_12 (Dense)             (None, 2)                 18
=================================================================
Total params: 604
Trainable params: 604
Non-trainable params: 0
_________________________________________________________________
None
>>>
>>> # 2. Compile
... model.compile(optimizer='adam',
...               loss='binary_crossentropy',
...               metrics=['accuracy'])
>>>
>>> # 3. Fit :
... model.fit(X_train, to_categorical(y_train), epochs = 10)
Epoch 1/10
  32/6833 [..............................] - ETA: 2:18 - loss: 1.3535 -  832/6833 [==>...........................] - ETA: 5s - loss: 0.8246 - ac1728/6833 [======>.......................] - ETA: 2s - loss: 0.7080 - ac2560/6833 [==========>...................] - ETA: 1s - loss: 0.6195 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.5378 - ac4288/6833 [=================>............] - ETA: 0s - loss: 0.4736 - ac4992/6833 [====================>.........] - ETA: 0s - loss: 0.4312 - ac5824/6833 [========================>.....] - ETA: 0s - loss: 0.3852 - ac6656/6833 [============================>.] - ETA: 0s - loss: 0.3473 - ac6833/6833 [==============================] - 1s 156us/step - loss: 0.3399 - acc: 0.8974
Epoch 2/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0299 - ac 832/6833 [==>...........................] - ETA: 0s - loss: 0.0650 - ac1536/6833 [=====>........................] - ETA: 0s - loss: 0.0736 - ac2304/6833 [=========>....................] - ETA: 0s - loss: 0.0661 - ac3136/6833 [============>.................] - ETA: 0s - loss: 0.0635 - ac3968/6833 [================>.............] - ETA: 0s - loss: 0.0613 - ac4896/6833 [====================>.........] - ETA: 0s - loss: 0.0614 - ac5824/6833 [========================>.....] - ETA: 0s - loss: 0.0579 - ac6560/6833 [===========================>..] - ETA: 0s - loss: 0.0569 - ac6833/6833 [==============================] - 0s 63us/step - loss: 0.0562 - acc: 0.9902
Epoch 3/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0080 - ac 896/6833 [==>...........................] - ETA: 0s - loss: 0.0449 - ac1792/6833 [======>.......................] - ETA: 0s - loss: 0.0431 - ac2592/6833 [==========>...................] - ETA: 0s - loss: 0.0457 - ac3456/6833 [==============>...............] - ETA: 0s - loss: 0.0444 - ac4224/6833 [=================>............] - ETA: 0s - loss: 0.0428 - ac5088/6833 [=====================>........] - ETA: 0s - loss: 0.0448 - ac6016/6833 [=========================>....] - ETA: 0s - loss: 0.0450 - ac6833/6833 [==============================] - 0s 59us/step - loss: 0.0435 - acc: 0.9902
Epoch 4/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0072 - ac 832/6833 [==>...........................] - ETA: 0s - loss: 0.0434 - ac1760/6833 [======>.......................] - ETA: 0s - loss: 0.0384 - ac2560/6833 [==========>...................] - ETA: 0s - loss: 0.0365 - ac3520/6833 [==============>...............] - ETA: 0s - loss: 0.0425 - ac4288/6833 [=================>............] - ETA: 0s - loss: 0.0414 - ac5088/6833 [=====================>........] - ETA: 0s - loss: 0.0409 - ac5920/6833 [========================>.....] - ETA: 0s - loss: 0.0396 - ac6833/6833 [==============================] - 0s 60us/step - loss: 0.0399 - acc: 0.9906
Epoch 5/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0075 - ac 800/6833 [==>...........................] - ETA: 0s - loss: 0.0386 - ac1728/6833 [======>.......................] - ETA: 0s - loss: 0.0476 - ac2496/6833 [=========>....................] - ETA: 0s - loss: 0.0456 - ac3328/6833 [=============>................] - ETA: 0s - loss: 0.0440 - ac3968/6833 [================>.............] - ETA: 0s - loss: 0.0426 - ac4832/6833 [====================>.........] - ETA: 0s - loss: 0.0399 - ac5600/6833 [=======================>......] - ETA: 0s - loss: 0.0387 - ac6272/6833 [==========================>...] - ETA: 0s - loss: 0.0391 - ac6833/6833 [==============================] - 0s 67us/step - loss: 0.0381 - acc: 0.9906
Epoch 6/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0046 - ac 704/6833 [==>...........................] - ETA: 0s - loss: 0.0114 - ac1248/6833 [====>.........................] - ETA: 0s - loss: 0.0216 - ac1824/6833 [=======>......................] - ETA: 0s - loss: 0.0209 - ac2560/6833 [==========>...................] - ETA: 0s - loss: 0.0208 - ac3296/6833 [=============>................] - ETA: 0s - loss: 0.0308 - ac4064/6833 [================>.............] - ETA: 0s - loss: 0.0351 - ac4896/6833 [====================>.........] - ETA: 0s - loss: 0.0324 - ac5728/6833 [========================>.....] - ETA: 0s - loss: 0.0354 - ac6496/6833 [===========================>..] - ETA: 0s - loss: 0.0383 - ac6833/6833 [==============================] - 0s 71us/step - loss: 0.0372 - acc: 0.9903
Epoch 7/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0168 - ac 896/6833 [==>...........................] - ETA: 0s - loss: 0.0285 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0341 - ac2528/6833 [==========>...................] - ETA: 0s - loss: 0.0388 - ac3232/6833 [=============>................] - ETA: 0s - loss: 0.0376 - ac4128/6833 [=================>............] - ETA: 0s - loss: 0.0360 - ac4832/6833 [====================>.........] - ETA: 0s - loss: 0.0391 - ac5696/6833 [========================>.....] - ETA: 0s - loss: 0.0391 - ac6528/6833 [===========================>..] - ETA: 0s - loss: 0.0366 - ac6833/6833 [==============================] - 0s 63us/step - loss: 0.0363 - acc: 0.9903
Epoch 8/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0049 - ac 896/6833 [==>...........................] - ETA: 0s - loss: 0.0271 - ac1568/6833 [=====>........................] - ETA: 0s - loss: 0.0330 - ac2464/6833 [=========>....................] - ETA: 0s - loss: 0.0316 - ac3264/6833 [=============>................] - ETA: 0s - loss: 0.0321 - ac4032/6833 [================>.............] - ETA: 0s - loss: 0.0348 - ac4768/6833 [===================>..........] - ETA: 0s - loss: 0.0342 - ac5568/6833 [=======================>......] - ETA: 0s - loss: 0.0332 - ac6240/6833 [==========================>...] - ETA: 0s - loss: 0.0335 - ac6833/6833 [==============================] - 0s 65us/step - loss: 0.0353 - acc: 0.9906
Epoch 9/10
  32/6833 [..............................] - ETA: 0s - loss: 0.0043 - ac 960/6833 [===>..........................] - ETA: 0s - loss: 0.0215 - ac1696/6833 [======>.......................] - ETA: 0s - loss: 0.0266 - ac2592/6833 [==========>...................] - ETA: 0s - loss: 0.0238 - ac3424/6833 [==============>...............] - ETA: 0s - loss: 0.0297 - ac4288/6833 [=================>............] - ETA: 0s - loss: 0.0316 - ac5120/6833 [=====================>........] - ETA: 0s - loss: 0.0352 - ac5984/6833 [=========================>....] - ETA: 0s - loss: 0.0344 - ac6752/6833 [============================>.] - ETA: 0s - loss: 0.0346 - ac6833/6833 [==============================] - 0s 61us/step - loss: 0.0342 - acc: 0.9906
Epoch 10/10
  32/6833 [..............................] - ETA: 0s - loss: 0.1445 - ac 800/6833 [==>...........................] - ETA: 0s - loss: 0.0557 - ac1664/6833 [======>.......................] - ETA: 0s - loss: 0.0465 - ac2432/6833 [=========>....................] - ETA: 0s - loss: 0.0424 - ac3328/6833 [=============>................] - ETA: 0s - loss: 0.0387 - ac4096/6833 [================>.............] - ETA: 0s - loss: 0.0371 - ac4864/6833 [====================>.........] - ETA: 0s - loss: 0.0347 - ac5760/6833 [========================>.....] - ETA: 0s - loss: 0.0359 - ac6656/6833 [============================>.] - ETA: 0s - loss: 0.0331 - ac6833/6833 [==============================] - 0s 62us/step - loss: 0.0338 - acc: 0.9903
<keras.callbacks.History object at 0x7fbd95de0668>
>>>
>>> # 5. Predict
... loss_and_metrics = model.evaluate(X_test, to_categorical(y_test))
2278/2278 [==============================] - 0s 78us/step
>>> loss_and_metrics
[0.048900732164671015, 0.98902546093064092]
>>> np.round(loss_and_metrics, 2)
array([ 0.05,  0.99])
>>> np.round(loss_and_metrics, 2)*100
array([  5.,  99.])
>>> print(np.round(loss_and_metrics, 2) * 100)
[  5.  99.]
>>> print(np.round(loss_and_metrics, 3))
[ 0.049  0.989]
>>> y_hat_nnet_pred = model.predict(X_test)
>>> np.round(y_hat_nnet_pred, 3) #here we see it can be [0.995, 0.005]
array([[ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       [ 1.        ,  0.        ],
       ...,
       [ 0.99599999,  0.004     ],
       [ 1.        ,  0.        ],
       [ 0.99800003,  0.002     ]], dtype=float32)
>>> # final predictions are the rounded version:
... y_hat_nnet = np.round(y_hat_nnet_pred, 1)
>>> np.round(y_hat_nnet_pred, 1)
array([[ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.],
       ...,
       [ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.]], dtype=float32)
>>> np.round(y_hat_nnet_pred, 0)
array([[ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.],
       ...,
       [ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.]], dtype=float32)
>>> np.round(0.5, 0)
0.0
>>> np.round(0.5, 1)
0.5
>>> np.round(0.5, 0)
0.0
>>> np.round(0.6, 0)
1.0
>>> np.round(0.51, 0)
1.0
>>> y_hat_nnet
array([[ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.],
       ...,
       [ 1.,  0.],
       [ 1.,  0.],
       [ 1.,  0.]], dtype=float32)
>>> raw_data.head()
                                   uuid  default  account_amount_added_12_24m  \
0  63f69b2c-8b1c-4740-b78d-52ed9a4515ac      0.0                            0
1  0e961183-8c15-4470-9a5e-07a1bd207661      0.0                            0
2  d8edaae6-4368-44e0-941e-8328f203e64e      0.0                            0
3  0095dfb6-a886-4e2a-b056-15ef45fdb0ef      0.0                            0
4  c8f8b835-5647-4506-bf15-49105d8af30b      0.0                            0

   account_days_in_dc_12_24m  account_days_in_rem_12_24m  \
0                        0.0                         0.0
1                        0.0                         0.0
2                        0.0                         0.0
3                        NaN                         NaN
4                        0.0                         0.0

   account_days_in_term_12_24m  account_incoming_debt_vs_paid_0_24m  \
0                          0.0                                  0.0
1                          0.0                                  NaN
2                          0.0                                  NaN
3                          NaN                                  NaN
4                          0.0                                  NaN

   account_status  account_worst_status_0_3m  account_worst_status_12_24m  \
0             1.0                        1.0                          NaN
1             1.0                        1.0                          1.0
2             NaN                        NaN                          NaN
3             NaN                        NaN                          NaN
4             NaN                        NaN                          NaN

            ...             status_3rd_last_archived_0_24m  \
0           ...                                          1
1           ...                                          1
2           ...                                          1
3           ...                                          1
4           ...                                          0

   status_max_archived_0_6_months  status_max_archived_0_12_months  \
0                               1                                1
1                               1                                2
2                               1                                2
3                               1                                1
4                               1                                1

   status_max_archived_0_24_months  recovery_debt  \
0                                1              0
1                                2              0
2                                2              0
3                                1              0
4                                1              0

  sum_capital_paid_account_0_12m sum_capital_paid_account_12_24m  \
0                              0                               0
1                              0                               0
2                              0                               0
3                              0                               0
4                              0                               0

   sum_paid_inv_0_12m  time_hours  worst_status_active_inv
0              178839    9.653333                      1.0
1               49014   13.181389                      NaN
2              124839   11.561944                      1.0
3              324676   15.751111                      1.0
4                7100   12.698611                      NaN

[5 rows x 43 columns]
>>> raw_data.head()
                                   uuid  default  account_amount_added_12_24m  \
0  63f69b2c-8b1c-4740-b78d-52ed9a4515ac      0.0                            0
1  0e961183-8c15-4470-9a5e-07a1bd207661      0.0                            0
2  d8edaae6-4368-44e0-941e-8328f203e64e      0.0                            0
3  0095dfb6-a886-4e2a-b056-15ef45fdb0ef      0.0                            0
4  c8f8b835-5647-4506-bf15-49105d8af30b      0.0                            0

   account_days_in_dc_12_24m  account_days_in_rem_12_24m  \
0                        0.0                         0.0
1                        0.0                         0.0
2                        0.0                         0.0
3                        NaN                         NaN
4                        0.0                         0.0

   account_days_in_term_12_24m  account_incoming_debt_vs_paid_0_24m  \
0                          0.0                                  0.0
1                          0.0                                  NaN
2                          0.0                                  NaN
3                          NaN                                  NaN
4                          0.0                                  NaN

   account_status  account_worst_status_0_3m  account_worst_status_12_24m  \
0             1.0                        1.0                          NaN
1             1.0                        1.0                          1.0
2             NaN                        NaN                          NaN
3             NaN                        NaN                          NaN
4             NaN                        NaN                          NaN

            ...             status_3rd_last_archived_0_24m  \
0           ...                                          1
1           ...                                          1
2           ...                                          1
3           ...                                          1
4           ...                                          0

   status_max_archived_0_6_months  status_max_archived_0_12_months  \
0                               1                                1
1                               1                                2
2                               1                                2
3                               1                                1
4                               1                                1

   status_max_archived_0_24_months  recovery_debt  \
0                                1              0
1                                2              0
2                                2              0
3                                1              0
4                                1              0

  sum_capital_paid_account_0_12m sum_capital_paid_account_12_24m  \
0                              0                               0
1                              0                               0
2                              0                               0
3                              0                               0
4                              0                               0

   sum_paid_inv_0_12m  time_hours  worst_status_active_inv
0              178839    9.653333                      1.0
1               49014   13.181389                      NaN
2              124839   11.561944                      1.0
3              324676   15.751111                      1.0
4                7100   12.698611                      NaN

[5 rows x 43 columns]
>>> vardescr
                               Variable         Type
0                                  uuid         text
1                               default  categorical
2           account_amount_added_12_24m      numeric
3             account_days_in_dc_12_24m      numeric
4            account_days_in_rem_12_24m      numeric
5           account_days_in_term_12_24m      numeric
6   account_incoming_debt_vs_paid_0_24m      numeric
7                        account_status  categorical
8             account_worst_status_0_3m  categorical
9           account_worst_status_12_24m  categorical
10            account_worst_status_3_6m  categorical
11           account_worst_status_6_12m  categorical
12                                  age      numeric
13               avg_payment_span_0_12m      numeric
14                avg_payment_span_0_3m      numeric
15                    merchant_category  categorical
16                       merchant_group  categorical
17                             has_paid      boolean
18                   max_paid_inv_0_12m      numeric
19                   max_paid_inv_0_24m      numeric
20                        name_in_email  categorical
21     num_active_div_by_paid_inv_0_12m      numeric
22                       num_active_inv      numeric
23                    num_arch_dc_0_12m      numeric
24                   num_arch_dc_12_24m      numeric
25                    num_arch_ok_0_12m      numeric
26                   num_arch_ok_12_24m      numeric
27                   num_arch_rem_0_12m      numeric
28           num_arch_written_off_0_12m      numeric
29          num_arch_written_off_12_24m      numeric
30                     num_unpaid_bills      numeric
31           status_last_archived_0_24m  categorical
32       status_2nd_last_archived_0_24m  categorical
33       status_3rd_last_archived_0_24m  categorical
34       status_max_archived_0_6_months  categorical
35      status_max_archived_0_12_months  categorical
36      status_max_archived_0_24_months  categorical
37                        recovery_debt      numeric
38       sum_capital_paid_account_0_12m      numeric
39      sum_capital_paid_account_12_24m      numeric
40                   sum_paid_inv_0_12m      numeric
41                           time_hours      numeric
42              worst_status_active_inv  categorical
>>>
>>>
>>>
>>> dfp = raw_data[pd.isnull(raw_data.default)]
>>> 0 == dfp.default.notnull().sum() #True
True
>>> Yp = dfp['default'] # Y for prediction
>>> Xp = dfp.drop('default', axis=1) # X for prediction
>>>
>>> # crate dfm: all rows that have deafult=1 or 0. this df is used for modeling.
... dfm = raw_data[pd.notnull(raw_data.default)]
>>> 0 == dfm.default.isnull().sum() #True: no NA in y
True
>>> 0 < dfm.isnull().sum().sum() #true: many NA in our X variables
True
>>>
>>> # Select Y for out model "Ym"
... Ym = dfm['default']
>>> Ym.mean() #concl: 0.014 so very few deafults
0.014314928425357873
>>>
>>> ## choice of X's and how to handle NA
...
>>> # a lot of NA in our X varibales. how should we handle that? qq
...
>>> # choice 1 is the one currently implemented
...
>>> # choice 1:
... # in the fitting step, ingore rows where any columns is NA (i.e. fit on "complete cases"). Then, in the predicting step, replace the NA in X with zero and use that data with the model to predict.
... # in addition, only use numerical X variables
...
>>> # choice 2:
... # NA handled as in choice 1
... # in addition, use numerical and categorical variables
...
>>> # choice 3:
... # replace NA with the column mean or median
... # the mean kan be imputed in the training set, and ev. also in the testing set.
...
>>> # choice 4:
... # replace NA with the "clustered mean". so first use some clustering algorithm to group observations together. then whenever there is an NA in row N column K look at the cluster that N belongs to, and impute that clusters mean value for kolumn K into the cell (N,K).
... # the mean kan be imputed in the training set, and ev. also in the testing set.
...
>>> # choice 5:
... # handle NA differently for different columns, based on our business knowledge. by looking at the names of the variables I see that some are related, e.g. account_status and account_worst_status (0-3m 12-24m )
...
>>> ## Drop NA rows
...
>>> # drop rows with NA. used for fitting :
... dfm = dfm.dropna()
>>> # set NA to 0 for X used in the prediction :
... Xp = Xp.fillna(0)
>>>
>>> ## Select X variables
...
>>> # Select numerical variables
... numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
>>> Xm = dfm[numerical_variables].as_matrix()
>>> Xp = dfp[numerical_variables].as_matrix()
>>> Ym.shape[0] == Xm.shape[0]  #check number of rows are the same
False
>>>
>>> # qq maybe increase to also include categorical
...
>>> ## Split into train and test
...
>>> # inside dfm we split into X_train and X_test  (70% and 30%) :
... X_train, X_test, y_train, y_test = train_test_split(Xm, Ym, random_state=9)
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py", line 2031, in train_test_split
    arrays = indexable(*arrays)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 229, in indexable
    check_consistent_length(*result)
  File "/home/jl/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py", line 204, in check_consistent_length
    " samples: %r" % [int(l) for l in lengths])
ValueError: Found input variables with inconsistent numbers of samples: [9111, 89976]
>>>
>>> # check shapes look ok. concl: size is enough for NN and KNN :
... X_train.shape, X_test.shape, Xp.shape
((6833, 25), (2278, 25), (10000, 25))
>>> y_train.shape, y_test.shape, Yp.shape
((6833,), (2278,), (10000,))
>>> (X_train.shape[0] + X_test.shape[0]) / len(raw_data) # concl: only 9% so maybe the data with NA is different from the data with non-NA. this would decrease the prediction quality of the model. if I have time, adress this.
0.09113187164919581
>>>
do2
