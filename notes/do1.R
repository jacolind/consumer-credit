## packages
library(dplyr)
#library(NeuralNetTools)
library(neuralnet)

## import
raw_data <- read.csv("dataset.csv", sep=";", header=T)
df <- raw_data

## check data

# na per column
apply(df, 2, function(x) sum(is.na(x)))
names(df)
sum(is.na(df['default']))
dim(df)

### analytics vidhy

## process data

# drop all NA observations
# qq

# select first 999 rows
data <- df[1:999, ]

# Random sampling
samplesize = 0.60 * nrow(data)
set.seed(80)
index = sample( seq_len ( nrow ( data ) ), size = samplesize )

# Create training and test set
datatrain = data[ index, ]
datatest = data[ -index, ]



## beckmw

library(NeuralNetTools)

# create model
library(neuralnet)
AND <- c(rep(0, 7), 1)
OR <- c(0, rep(1, 7))
binary_data <- data.frame(expand.grid(c(0, 1), c(0, 1), c(0, 1)), AND, OR)
mod <- neuralnet(AND + OR ~ Var1 + Var2 + Var3, binary_data,
                 hidden = c(6, 12, 8), rep = 10, err.fct = 'ce', linear.output = FALSE)
