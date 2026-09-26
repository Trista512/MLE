import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# process silver table for financials: clean, cast schema, and derive features
def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # load bronze partition
    partition_name = "bronze_feature_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # helper: strip everything except digits, dot, minus -> then cast
    def clean_numeric(colname, cast_type):
        return F.regexp_replace(col(colname).cast(StringType()), r'[^0-9.\-]', '').cast(cast_type)

    # clean numeric columns polluted with underscores / junk
    df = df.withColumn("Annual_Income", clean_numeric("Annual_Income", FloatType()))
    df = df.withColumn("Num_of_Loan", clean_numeric("Num_of_Loan", IntegerType()))
    df = df.withColumn("Num_of_Delayed_Payment", clean_numeric("Num_of_Delayed_Payment", IntegerType()))
    df = df.withColumn("Outstanding_Debt", clean_numeric("Outstanding_Debt", FloatType()))
    df = df.withColumn("Amount_invested_monthly", clean_numeric("Amount_invested_monthly", FloatType()))
    df = df.withColumn("Monthly_Balance", clean_numeric("Monthly_Balance", FloatType()))
    df = df.withColumn("Changed_Credit_Limit", clean_numeric("Changed_Credit_Limit", FloatType()))

    # cast already-clean numeric columns to enforce schema
    df = df.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))
    df = df.withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Card", col("Num_Credit_Card").cast(IntegerType()))
    df = df.withColumn("Interest_Rate", col("Interest_Rate").cast(IntegerType()))
    df = df.withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast(IntegerType()))
    df = df.withColumn("Credit_Utilization_Ratio", col("Credit_Utilization_Ratio").cast(FloatType()))
    df = df.withColumn("Total_EMI_per_month", col("Total_EMI_per_month").cast(FloatType()))

    # parse Credit_History_Age "X Years and Y Months" -> total months
    df = df.withColumn("Credit_History_Age_months",
                       (F.regexp_extract(col("Credit_History_Age"), r'(\d+)\s*Years?', 1).cast(IntegerType()) * 12
                        + F.regexp_extract(col("Credit_History_Age"), r'(\d+)\s*Months?', 1).cast(IntegerType())))

    # Type_of_Loan -> number of loan types (feature engineering)
    df = df.withColumn("Num_loan_types",
                       F.when(col("Type_of_Loan").isNull(), 0)
                       .otherwise(F.size(F.split(F.regexp_replace(col("Type_of_Loan"), " and ", ","), ","))))

    # clean categorical placeholders -> null
    df = df.withColumn("Credit_Mix",
                       F.when(col("Credit_Mix").isin("Bad", "Standard", "Good"), col("Credit_Mix")).otherwise(None))
    df = df.withColumn("Payment_of_Min_Amount",
                       F.when(col("Payment_of_Min_Amount").isin("Yes", "No", "NM"), col("Payment_of_Min_Amount")).otherwise(None))
    df = df.withColumn("Payment_Behaviour",
                       F.when(col("Payment_Behaviour").isin(
                           "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
                           "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
                           "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"),
                           col("Payment_Behaviour")).otherwise(None))

    # clip unrealistic outliers
    df = df.withColumn("Num_of_Loan", F.when(col("Num_of_Loan") < 0, 0).when(col("Num_of_Loan") > 50, 50).otherwise(col("Num_of_Loan")))
    df = df.withColumn("Num_of_Delayed_Payment", F.when(col("Num_of_Delayed_Payment") < 0, 0).when(col("Num_of_Delayed_Payment") > 100, 100).otherwise(col("Num_of_Delayed_Payment")))
    df = df.withColumn("Num_Bank_Accounts", F.when(col("Num_Bank_Accounts") < 0, 0).when(col("Num_Bank_Accounts") > 20, 20).otherwise(col("Num_Bank_Accounts")))
    df = df.withColumn("Monthly_Balance", F.when((col("Monthly_Balance") < -100000) | (col("Monthly_Balance") > 1000000), None).otherwise(col("Monthly_Balance")))

    # enforce key column types
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # drop raw columns replaced by engineered ones
    df = df.drop("Credit_History_Age", "Type_of_Loan")

    # save silver table
    partition_name = "silver_feature_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


# process silver table for clickstream: enforce schema (already clean)
def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # load bronze partition
    partition_name = "bronze_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # enforce schema: fe_1..fe_20 as integer, keys typed
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table
    partition_name = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df