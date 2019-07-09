/* 
For this assessment you must submit your work as a report – there will be
questions of interest that you must answer and discuss by conducting a
statistical analysis in SAS. Start your report with a brief introduction to the
problem, clearly outlining the motivation for your analysis (i.e. discuss what
the questions of interest are). Next do an exploratory analysis for each
question of interest a) to c) (no formal statistical tests or models at this
point) – this might include plotting your data, looking at summary statistics
etc in order to get an initial impression as to the conclusions of your
questions of interest.

Following this, you will answer each question of interest a) to i) with a formal
statistical analysis. Finally, give a conclusion summarising the overall results
of your report. Since this is a lab report, ensure that any figures/tables etc
are clearly explained/discussed and justify any of your conclusions. Do not
leave it to the reader to hunt for information (walk them through any output or
conclusion you make), you will be marked on the clarity of your report. You can
either split the report into exploratory analysis and formal analysis or present
the exploratory analysis for each of a)-c) alongside the formal analysis for
each of a)-c).

For each question of interest, you will submit two things:  1) Relevant SAS
output. This might include tables and plots from SAS containing the relevant
results of whatever code you have written. Make sure your output is clearly
titled, labelled etc. This should be done within your SAS code – do not edit
this with other software.  2) Discussion of whatever analysis you have
conducted.

/* 
Questions of interest f) and i) ask you to additionally submit your code –
this will be clearly outlined in the question of interest, do not submit code
for any other question of interest.  
*/
libname input "/courses/dc36fc35ba27fe300/Assessment";
libname all "/courses/dc36fc35ba27fe300";
libname out "/home/v0008/sasuser.v94/datasets";

/* 
Question A 
A temperature for this country of 25.C or above can alert
officials to the possibility of a drought. Is the mean maximum temperature
different to this? 
*/
proc univariate data = input.weather plots;
	var MaxTemp;
run;

proc ttest data = input.weather H0 = 25 plots(shownull) = interval;
	var MaxTemp;
run;

proc ttest data = input.weather H0 = 25 plots(shownull) = all;
	var MaxTemp;
run;

/* 
Question B 
Investigate whether there is a difference between the mean maximum
temperature and the mean minimum temperature. 
*/
proc ttest data = input.weather;
	paired MaxTemp * MinTemp;
run;

/* 
Question C 
It is quite possible that evaporation is associated with the
strongest wind gust direction for a day. Check whether there is a difference in
the mean evaporation between the different strongest wind gust directions using
an appropriate multiple comparison adjustment. 
*/

* We first produce diagnostic plots to check normality assumption;
proc glm data = input.weather plots = diagnostics;
	class WindGustDir;
	model Evaporation = WindGustDir;
	means WindGustDir;
run;

* Then we perform a Levene's test to check equal variances assumption;
proc glm data = input.weather plots = diagnostics;
	class WindGustDir;
	model Evaporation = WindGustDir;
	means WindGustDir / hovtest=levene;
run;

* We perform post-hoc analysis without asjusted p-values first;
proc glm data = input.weather plots(only)=(diffplot(center)); 
	class WindGustDir;
	model Evaporation = WindGustDir;
	lsmeans WindGustDir / pdiff = all adjust = T;
run; 

* Post-hoc using Tukey method;
proc glm data = weather_combined plots(only)=(diffplot(center)); 
	class WindGustDirCombined;
	model Evaporation = WindGustDirCombined;
	lsmeans WindGustDirCombined / pdiff = all adjust = tukey;
run;

proc anova data = input.weather;
	class WindGustDir;
	model Evaporation = WindGustDir;
	means WindGustDir / tukey;
run;

/* 
Question D 
Find the “best” model (as determined by an automated model
selection procedure), using evaporation as the response and all other variables
in the dataset as potential predictors (do not consider any interaction terms,
transformations or higher order  terms, and do not include the variable
“RainTomorrow” in the modelling process).

When you come to fit a model for this question of interest, use two different
directional automated procedures (fit two different models considering the same
proposed predictors, using different directional selection approaches) and
examine whether they result in the same or different final models. If they are
different, which do you prefer? Justify your answer. 
*/
%let variables = MinTemp MaxTemp Rainfall Sunshine WindGustSpeed WindSpeed9am WindSpeed3pm Humidity9am Humidity3pm Pressure9am  Pressure3pm Temp9am Temp3pm;
%let categories = Location NewEquipment WindGustDir WindDir9am WindDir3pm Status RainToday Cloud3pm Cloud9am;

proc glmselect data = input.weather plots=all;
	class &categories;
	model Evaporation = &variables / selection = backward select=SBC slstay=0.05 showpvalues;
run;


/* 
Question E 
Since the variables “WindGustDir”, “WindDir9am” and “WindDir3pm”
have many levels, it is of interest to see whether some levels could be
combined. Perform Greenacre’s method, with the variable “RainTomorrow” as the
response, for each of these variables in turn. If, for any of the above
variables, Greenacre’s method suggests combining levels is reasonable, combine
the appropriate levels and use this(these) new variable(s) in any subsequent
analysis (give any new variables new names). 
*
* https://vimeo.com/301775681
* http://www.econ.upf.edu/~michael/stanford/maeb7.pdf
*/
data weather;
	set input.weather;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

proc means data=weather noprint nway;
	class WindGustDir;
	var RainTomorrowBinary;
	output out=greenacre01 mean=prop;
run;

ods output clusterhistory=weather;

proc cluster data=greenacre01 method=ward 
	plots=(dendrogram(vertical height=rsq));
	freq _freq_;
	var prop;
	id WindGustDir;
run;

*.....;

data weather;
	set input.weather;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

proc means data=weather noprint nway;
	class WindDir9am;
	var RainTomorrowBinary;
	output out=greenacre02 mean=prop;
run;

ods output clusterhistory=weather;

proc cluster data=greenacre02 method=ward 
	plots=(dendrogram(vertical height=rsq));
	freq _freq_;
	var prop;
	id WindDir9am;
run;

*.....;

data weather;
	set input.weather;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

proc means data=weather noprint nway;
	class WindDir3pm;
	var RainTomorrowBinary;
	output out=greenacre03 mean=prop;
run;

ods output clusterhistory=weather;

proc cluster data=greenacre03 method=ward 
	plots=(dendrogram(vertical height=rsq));
	freq _freq_;
	var prop;
	id WindDir3pm;
run;

*.....;

data work.weather_combined;
	set input.weather;
	
	if WindGustDir in ("E" "NNE" "ESE" "SW" "NE" "ENE")
		then WindGustDirCombined = "Cluster1";
	else if WindGustDir in ("S" "SE" "SSE")
		then WindGustDirCombined = "Cluster1";
	else WindGustDirCombined = "Cluster2"; 
	
	if WindDir9am in ("E" "ENE" "SE" "ESE")
		then WindDir9amCombined = "Cluster1";
	else if WindDir9am in ("N" "NNW" "WNW")
		then WindDir9amCombined = "Cluster2";
	else WindDir9amCombined = "Cluster3";
	
	if WindDir3pm in ("E" "ENE" "NE" "SE" "S" "SSW" "WSW")
		then WindDir3pmCombined = "Cluster1";
	else if WindDir3pm in ("ESE" "NNE" "SSE" "SW")
		then WindDir3pmCombined = "Cluster1";
	else WindDir3pmCombined = "Cluster2";
run;

*.....;

data weather;
	set work.weather_combined;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

proc means data=weather noprint nway;
	class WindGustDirCombined;
	var RainTomorrowBinary;
	output out=greenacre01 mean=prop;
run;

ods output clusterhistory=weather;

proc cluster data=greenacre01 method=ward 
	plots=(dendrogram(vertical height=rsq));
	freq _freq_;
	var prop;
	id WindGustDirCombined;
run;

*.....;

data weather;
	set work.weather_combined;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

proc means data=weather noprint nway;
	class WindDir9amCombined;
	var RainTomorrowBinary;
	output out=greenacre02 mean=prop;
run;

ods output clusterhistory=weather;

proc cluster data=greenacre02 method=ward 
	plots=(dendrogram(vertical height=rsq));
	freq _freq_;
	var prop;
	id WindDir9amCombined;
run;

*.....;

data weather;
	set work.weather_combined;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

proc means data=weather noprint nway;
	class WindDir3pmCombined;
	var RainTomorrowBinary;
	output out=greenacre03 mean=prop;
run;

ods output clusterhistory=weather;

proc cluster data=greenacre03 method=ward 
	plots=(dendrogram(vertical height=rsq));
	freq _freq_;
	var prop;
	id WindDir3pmCombined;
run;

/* 
Question F 
Split your data for model assessment into training data (75% of
original dataset), and test data (25% of original dataset), ensuring you have a
roughly equal proportion of Yes to No responses for the variable “RainTomorrow”
in each dataset. In addition to output and discussion, include your code in your
report for this question of Interest. 
*/
data weather;
	set work.weather_combined;
run;

proc sort data=weather out=work.weather_sort;
	by RainTomorrow;
run;

proc surveyselect noprint data=weather_sort samprate=.75 outall out=weather_sampling;
	strata RainTomorrow;
run;

data weather_training(drop=selected SelectionProb SamplingWeight) 
	 weather_test(drop=selected SelectionProb SamplingWeight);
	set work.weather_sampling;
	if selected then output weather_training;
	else output weather_test;
run;

*76.90 percentage of NO;
proc freq data=work.weather_training;
   tables RainTomorrow;
run;

*76.96 percentage of NO;
proc freq data=work.weather_test;
   tables RainTomorrow;
run;

/* 
Question G 
Using Hoeffding’s and Spearman’s statistics, check whether there
is any evidence of non-linearity and/or irrelevant variables for the continuous
variables in your dataset (use the training data).
*/

%let variables = MinTemp MaxTemp Rainfall Sunshine WindGustSpeed WindSpeed9am WindSpeed3pm Humidity9am Humidity3pm Pressure9am Pressure3pm Temp9am Temp3pm;

ods output spearmancorr=work.spearman hoeffdingcorr=work.hoeffding;

proc corr data=work.weather_training spearman hoeffding;
	var Evaporation;
	with &variables;
run;

proc sort data=work.spearman;
	by variable;
run;

proc sort data=work.hoeffding;
	by variable;
run;

data work.coefficients;
	merge work.spearman(rename=(Evaporation=scoef pEvaporation=spvalue))
		work.hoeffding(rename=(Evaporation=hcoef pEvaporation=hpvalue));
	by variable;
	scoef_abs=abs(scoef);
	hcoef_abs=abs(hcoef);
run;

proc rank data=work.coefficients out= work.coefficients_rank;
	var scoef_abs hcoef_abs;
	ranks ranksp rankho;
run;

proc print data=work.coefficients_rank;
	var variable ranksp rankho scoef spvalue hcoef hpvalue;
run;

proc sgplot data=work.coefficients_rank;
	scatter y=ranksp x=rankho / datalabel=variable;
run;

/* 
Question H 
Find the “best” model (as determined by an automated model selection procedure), 
using the variable “RainTomorrow” as the response and all other variables in the 
dataset as potential predictors. Use the training data and do not consider any 
interaction terms, transformations or higher order terms. 
*/
%let variables = MinTemp MaxTemp Rainfall Sunshine WindGustSpeed WindSpeed9am WindSpeed3pm Humidity9am Humidity3pm Pressure9am  Pressure3pm Temp9am Temp3pm Evaporation;
%let variables = MinTemp MaxTemp Sunshine WindGustSpeed Humidity9am Humidity3pm Temp9am Temp3pm Evaporation;
%let categories = Location NewEquipment WindGustDirCombined WindDir9amCombined WindDir3pmCombined Status RainToday Cloud3pm Cloud9am;

data work.weather_training;
	set work.weather_training;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

*0.1 for backwards, 0.5 for forward, 0.15  for stepwise model selection;
*slstay, slentry;
proc glmselect data=work.weather_training;
	class &categories;
	model RainTomorrowBinary = &variables &categories / selection=stepwise select=AIC showpvalues;
run;

/* 
Question I 
It is of interest to see whether a specific subset of the
variables can be used to make accurate predictions about whether it will rain
tomorrow. Consider the model that only uses the variables “RainToday” and
“MaxTemp” as predictors. Investigate how the level of predictive performance to
the test data differs using this model to that of your final model from part h).
In addition to output and discussion, include your code in your report for this
question of interest. 
*/
data work.weather_training;
	set work.weather_training;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;  
	else RainTomorrowBinary = 1;
run;

data work.weather_test;
	set work.weather_test;
	if RainTomorrow = "No" then RainTomorrowBinary = 0;
	else RainTomorrowBinary = 1;
run;

proc logistic data=work.weather_training;
	class RainToday  ;
	model RainTomorrowBinary(event='1')=MinTemp MaxTemp Sunshine WindGustSpeed Humidity9am Humidity3pm WindDir3pmCombined RainToday Cloud3pm;
	score data=work.weather_test out=testAssess(rename=(p_1=p_complex)) outroc=work.roc;
run;

proc logistic data=work.weather_training;
	class RainToday;
	model RainTomorrowBinary(event='1')=RainToday MaxTemp;
	score data=work.testAssess out=testAssess(rename=(p_1=p_simple)) outroc=work.roc;
run;

proc logistic data=work.testAssess;
	model RainTomorrowBinary(event='1')=p_complex p_simple/nofit;
	roc "Complex Model" p_complex;
	roc "Simple Model" p_simple;
	roccontrast "Comparing Models";
run;
