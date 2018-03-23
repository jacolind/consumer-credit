# see cpde snipped on combined_set


### Clean data
################################################################################

# create dfp: all rows that have deafult=NA. this df is used for prediction.
dfp = raw_data[pd.isnull(raw_data.default)]
# crate dfm: all rows that have deafult=1 or 0. this df is used for modeling.
dfm = raw_data[pd.notnull(raw_data.default)]

## choice of X's and how to handle NA

# how many NA?
total_na = dfm.isnull().sum().sum()
total_cells =  dfm.count().sum()
total_na / total_cells * 100

# handle NA in dfm :
dfm = dfm.fillna(0) # fill NA with zero

# Select y for prediction and modeling
yp = dfp['default']
ym = dfm['default']
ym.mean() #concl: 0.014 so very few deafults



## handle categorical variables "objects"

# combine sets
combined_set = pd.concat([dfm, dfp], axis = 0)
# Loop through all columns in the dataframe, convert categorical to integer
for feature in combined_set.columns:
    if combined_set[feature].dtype == 'object':
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes
# uncombine sets
dfp = combined_set[dfm.shape[0]:]
dfm = combined_set[:dfm.shape[0]]


## Select X variables

exclude = ['default', 'uuid']
Xm = dfm.drop(exclude, axis=1).fillna(0)
Xp = dfp.drop(exclude, axis=1).fillna(0)
ym.shape[0] == Xm.shape[0]
Xm.shape

## standardize X

scaler = StandardScaler().fit(Xm)
Xm = scaler.transform(Xm)
Xp = scaler.transform(Xp)

a
