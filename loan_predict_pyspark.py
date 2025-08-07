#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import when
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Define base directory

base_dir = os.path.abspath(os.path.dirname(__file__))

# Construct full input and output paths
input_path = f"file://{os.path.join(base_dir, 'loan_data.csv')}"
output_path = os.path.join(base_dir, "output.txt")

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("LoanPredictML").getOrCreate()

# Step 2: Load dataset
input_path="https://raw.githubusercontent.com/learningtechnologieslab/mds_cloud_computing/refs/heads/main/apache_spark/loan_data.csv"
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Step 3: Drop rows with missing values
# Most of the time when filling out a loan application there are mandatory fields. 
# An incomplete application may be invalid or not submitted correctly.
# These incomplete values can introduce noise to the analysis so my decision is to drop them from the data set.

df=df.dropna()

# Step 4 Removing features and justifications
# The following features will be removed: Loan_ID, Gender, Married
# The features are removed due to the fact that discriminating based on them is in direct violation of the Equal -
# Credit Opportunity Act (ECOA)
# Loan_ID is dropped because it is just a unique identifer.

df = df.drop("Loan_ID","Gender","Married")

# Features that are kept
# Dependents: There is an innate value with having to take care of dependents. (Other financial obligations)
# Education: Education typically is a good indicator of a persons income. Educated people tend to have higher income
# Self Employed: Typically if you are self employed there are other financial obligations with being self employed-
# like business expense
# ApplicantIncome: Important, you want to know how much a person makes to evaluate a comfortable loan amount
# CoapplicantIncome: They generate income so they could also help with the financial burden
# Loan_amount_term: If you have a longer loan you may be able to get more money, compared to a shorter loan. Depends on income
# Credit History: How likely you are to repair the loan
# Property_Area: In case you default on the loan, what is the best chance they have to recoup the value you defaulted on.-
# They would sell the property. Whatever amount is left over from the sale you would owe the lending company.



# Step 5: Address Outliers
# I chose not to remove outlier from the dataset since applicants come from a wide variety of backgrounds. 
# In the real word, there are individuals with low incomes and high incomes that apply for loans.
# Although the salary distribution is skew-right these outlier represent an important part of our population.
# Excluding them could lead to a biased model and reduce the generalizability of the results.
# Keeping all of the data points helps the model better reflect the diversity of the applicants in a real life situation.

# Step 6: Discretizing the following categories: Applicantincome, coapplicantincome, loanamount, Loan_amount_term

# Applicant/Coapplicantincome : Low (<3000), Middle(3000-6000), High(>6000) 
# Low income is anything lower than 3000 as anything lower than 3000 is low compared to other values.
# Middle income is anything between 3000 and 6000 as that what I'm seeing as an average range
# High Income: Anything over 6000 is high income

df = df.withColumn("ApplicantIncomeDis",
    when(df["ApplicantIncome"] < 3000, "Low")
    .when((df["ApplicantIncome"] >= 3000) & (df["ApplicantIncome"] <= 6000), "Middle")
    .otherwise("High")
)

df = df.withColumn("CoapplicantIncomeDis",
    when(df["CoapplicantIncome"] < 3000, "Low")
    .when((df["CoapplicantIncome"] >= 3000) & (df["CoapplicantIncome"] <= 6000), "Middle")
    .otherwise("High")
)


# Loanamount: Small (<100), Medium (100-200), Large (>200)
# A small loan is anything less than 100 (small loan request)
# Medium loan is anything between 100-200 (standard Loan)
# Large Loan is anything greater than 200 (Higher financial need)

df=df.withColumn("LoanAmountDis",
    when(df["LoanAmount"] < 100, "Small")
    .when((df["LoanAmount"] >= 100) & (df["LoanAmount"] <= 200), "Medium")
    .otherwise("Large")
)

# Loan_amount_term: Short(<180), medium(180-300),Long(>300)
# Short term is less than 180 months or 15 years
# Medium is 180-300 months (15-25 Years)
# Long term is anything greater than 300 months

df=df.withColumn("LoanAmountTermDis",
    when(df["Loan_Amount_Term"] < 180, "Short")
    .when((df["Loan_Amount_Term"] >= 180) & (df["Loan_Amount_Term"] <= 300), "Medium")
    .otherwise("Long")
)

# Step 7: Encode categorical variables

dependents_indexer= StringIndexer(inputCol="Dependents", outputCol="Dependents_Indexed")
education_indexer= StringIndexer(inputCol="Education", outputCol="Education_Indexed")
self_employed_indexer= StringIndexer(inputCol="Self_Employed", outputCol="Self_Employed_Indexed")
property_area_indexer=StringIndexer(inputCol="Property_Area", outputCol="Property_Area_Indexed")
loan_status_indexer=StringIndexer(inputCol="Loan_Status", outputCol="Loan_Status_Indexed")

# Step 8: Assemble Features
assembler=VectorAssembler(
    inputCols=["Dependents_Indexed","Education_Indexed","Self_Employed_Indexed","ApplicantIncome", "CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area_Indexed"],
    outputCol="features"
)

# Step 9: Classifiers

classifiers = {
"RandomForest":RandomForestClassifier(labelCol="Loan_Status_Indexed", featuresCol="features",numTrees=100),
"DecisionTree":DecisionTreeClassifier(labelCol="Loan_Status_Indexed", featuresCol="features", maxDepth=5),
"NaiveBayes":NaiveBayes(labelCol="Loan_Status_Indexed", featuresCol="features",modelType="multinomial")
}
    
# Step 10 Build ML Pipeline
results = []

for name, c in classifiers.items():
    pipeline= Pipeline(stages=[
        loan_status_indexer,
        dependents_indexer,
        education_indexer,
        self_employed_indexer,
        property_area_indexer,
        assembler,
        c
    ])

    # Step 11: Split into Train/Test
    train_data, test_data = df.randomSplit([0.8,0.2], seed=42)

    # Step 12: Train Model

    model=pipeline.fit(train_data)

    # Step 13: Make predictions
    predictions=model.transform(test_data)

    # Step 14: Evaluate Model
    evaluator= MulticlassClassificationEvaluator(labelCol='Loan_Status_Indexed', predictionCol="prediction", metricName="accuracy")
    accuracy=evaluator.evaluate(predictions)
    results.append(f"{name} Accuracy: {accuracy:.4f}")
    
# Step 15: Save output
with open(output_path, "w") as f:
    f.write("\n".join(results))

# Step 16 stopping spark
spark.stop()

