# 20171129

default1_20171129 had a certain setup. 

split into dfm and dfp for model and predict. fill na with 0. standardize variables. 

dfm has Xm and ym. dfp has Xp and yp, they have 10000 rows and all yp=NaN.

on dfm fit three models, using cv=5. the one with highest auc is the winning model. 

apply the winning model on Xp to predict proba for yp.

this is in the file named `default1_20171129`

# 20171215 

i changed this to another method: 

* dela upp dataset i Xf Xt Xp för fit tune predict. där Xf+Xt är gamla Xm.  lagom kanske är Xf/Xt på 80/20
* använd cv=5 i Xf och se vilka som är bäst med avseende på AUC. har redan gjort detta. välj den modelltyp (tex logreg) samt dess best params och spara "winnermodel". det är ett "fitobject". nu är modellen bestämd. 
* använd winndermodel-fit för att på Xt (dvs ej refit) för att predicta proba. skapa lista med thresholds. plotta upp classification matrix så som glrs i In[17]. välj den threshold som ger en bra recall - enligt din subjektiva uppfattning. infoga kommentar om kostnad för FP och FN kan implementeras men jag känner ej till sådan kostnad. nu är threshold bestämd "winenrthreshold"
* nu har du winnermodel och winnerthreshold. använd spec av winnermodel för att göra en refit med hela Xm. gör en predict_proba med denna modell utav Xp. konvertera dessa proba till classes med den bestämda winnerthreshold. spara proba och classes. 



# 
