# Information Retrieval (M) 2018-2019

Project about how learning-to-rank can improve information retrieval effectiveness. Two proximity features were implemented, tested and analyzed using [Terrier](http://terrier.org/docs/current/learning.html).

Final report can be found [here](2419105v.pdf). It is generated from a [.Rmd file](2419105v.Rmd). Data to generate the report from the `.Rmd` file is not provided because of Copyright. Source code (Java) to implement the features can be found [here](terrier). 

--------------------

# Exercise 2

## Introduction
Learning-to-rank is a recent paradigm used by commercial search engines to improve retrieval effectiveness by combining different sources of evidence (aka features). The key point of learningto-rank is that it is easy to incorporate new features and to leverage the amount of potential training data available to Web search engines. In this exercise, you will be trying learning to rank using a number of standard or provided features, and implementing two additional features of your own.

In particular, you will be implementing, testing and evaluating two proximity features, which aim to boost the scores of documents where the query terms appear in close proximity. Effectiveness will be measured using two metrics: Mean Average Precision (MAP) and Precision at Rank 5 (P@5).

## Exercise Specification
For retrieval, you will use Terrier’s support for learning-to-rank (LTR) and the included state-ofthe-art Jforests LambdaMART LTR technique. In particular, the online documentation about LTR available at http://terrier.org/docs/current/learning.html will be an essential guide on how to conduct your experiments. However, you will only be using a limited set of 3 features as listed
below. 

```
WMODEL:SingleFieldModel(BM25,0)
QI:StaticFeature(OIS,/users/level4/software/IR/Resources/features/pagerank.oos.gz)
DSM:org.terrier.matching.dsms.DFRDependenceScoreModifier
```

Your first task is to deploy and evaluate a baseline LTR approach using the provided 3 features following the Terrier LTR instructions mentioned above. For generating the required LTR *sample*, you need to use the PL2 weighting model. The sample is then re-ranked using the 3 provided features.

### Question 1
First, in a table, report the effectiveness performances of the two system configurations: LTR (using PL2 to generate the sample) vs. PL2. In your report, show the results in a table as follows (Report all your MAP and P@5 performances to 4 decimal places): 

Use the t-test statistical significance test to conclude if the performances of LTR are significantly better than that of PL2 using MAP or P@5. For example, to conduct a t-test, you can use the online t-test tool available at:
http://www.socscistatistics.com/tests/studentttest/ and select a two-tailed test with a significance level of 0.05.

For your two statistical tests, report their outcomes using the following format:

`LTR vs PL2 on Metric M (M being either MAP or P@5): The t-value is XXXX. The pvalue is XXXX. The result is <significant/not significant> at p < .05.`

Now, using the outcome of your t-test tool, state if LTR (PL2 sample) is better by a statistically significant margin than PL2 on either or both used effectiveness metrics.

### Question 2
Now, you should implement two additional proximity search features – a proximity search feature allows documents where the query terms occur closely together to be boosted. We require that you implement two of the functions numbered 1-5 in the following paper:

```
Ronan Cummins and Colm O'Riordan. 2009. Learning in a pairwise term-term proximity
framework for information retrieval. In Proceedings of ACM SIGIR 2009.
http://ir.dcs.gla.ac.uk/~ronanc/papers/cumminsSIGIR09.pdf
```

**NB:** You should calculate your feature by aggregating the function score (mean or min or max, as appropriate) over all pairs of query terms.

You will implement your new features as two `DocumentScoreModifiers (DSM)` classes, using the example DSM code provided in the Github repository (https://github.com/cmacdonald/IRcourseHM).

You can add a DSM as an additional feature by appending its full name to the feature file, e.g.:

```
DSM:org.myclass.MyProx1DSM
```

**Q2a.** Name the two proximity features you have chosen to implement and provide a brief rationale for your choice of these two particular features, especially in terms of how they might affect the performance of the deployed LTR baseline approach of Q1.

**Q2b.** Along with the submission of your source code, discuss briefly your implementation of the two features, highlighting in particular any assumptions or design choices you made, any difficulties that you had to overcome to implement the two features, and how these difficulties were reflected in the unit testing you conducted.

**Q2c.** Along with the submission of the source code of your unit tests, describe the unit tests that you have conducted to test that your implemented features behaved as expected. In particular, highlight any specific cases you tested your code for, and whether you identified and corrected any error in your code.

### Question 3
Once you have created and tested your two DSMs, you should experiment with LTR, including your new features, and comparing to your LTR baseline with the 3 initial provided features. As per best evaluation practices, when determining the benefits of your added features, you should add them separately to the list of provided features, then in combination to see if they provide different sources of evidence to the learner. This gives you 4 settings to compare and discuss (LTR baseline,
LTR Baseline + DSM 1, LTR Baseline + DSM 2, LTR Baseline + DSM 1 + DSM 2). Report the obtained performances of your 4 LTR system variants in a table

Using the t-test statistical test, state if the introduction of either of your proximity features (or both) has led to a statistically significant enhancement of your LTR (baseline) performance *in terms of MAP* (i.e. you will conduct 3 tests in total). Report the details of your significance tests as in Q1.

### Question 4
Using MAP as the main evaluation metric, provide a concise, yet informative, discussion on the performance of the learned model with and without your additional features. In particular, comment on why your features did/did not help (individually or in combination), and what queries benefitted (their numbers, their nature and characteristics, etc.).

As part of your discussion, you will need to provide and use the following:

a) A recall-precision graph summarising the results of your 4 LTR system variants,

b) A histogram with a query-by-query performance analysis of the 4 system variants used.

c) A suitable table summarising the number of queries that have been improved/degraded/unaffected with the introduction of either (or both) of your proximity
features with respect to the LTR baseline.

d) A suitable table showing examples of queries that have been particularly improved or harmed with the introduction of your proximity search features.

### Question 5
An overall reflection about what you have learnt from conducting the experiments and what your main conclusions are from this exercise.

## Hand-in Instructions:
(a) Submit a PDF report (5 pages MAX) with your answers to Q1-Q5.

(b) Submit the source code of your implemented features and their unit tests.

Please make your submission on the Moodle Exercise 2 Submission instance. This exercise is worth 50 marks (48 marks + 2 marks for the quality of presentation of the report) and 12% of the final course grade.