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

from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# process gold feature store: combine selected features from clickstream and financials (point-in-time correct)
def process_gold_feature_store(snapshot_date_str, silver_clickstream_directory, silver_financials_directory, gold_feature_store_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # selected features from EDA (|correlation with default| >= 0.10)
    selected_clickstream = ["fe_5", "fe_10"]
    selected_financials = ["Delay_from_due_date", "Outstanding_Debt", "Credit_Mix",
                           "Credit_History_Age_months", "Num_loan_types", "Payment_of_Min_Amount",
                           "Num_Bank_Accounts", "Num_of_Loan", "Monthly_Inhand_Salary",
                           "Monthly_Balance", "Num_of_Delayed_Payment"]

    # load clickstream silver partition for this month (no future months in any row)
    cs_partition = "silver_feature_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    cs = spark.read.parquet(silver_clickstream_directory + cs_partition)
    cs = cs.select(["Customer_ID", "snapshot_date"] + selected_clickstream)

    # load financials silver, point-in-time correct: only records known on/before this snapshot_date
    fin = spark.read.parquet(silver_financials_directory + "*.parquet")
    fin = fin.filter(col("snapshot_date") <= snapshot_date)
    w = Window.partitionBy("Customer_ID").orderBy(F.desc("snapshot_date"))
    fin = fin.withColumn("rn", F.row_number().over(w)).filter(col("rn") == 1)
    fin = fin.select(["Customer_ID"] + selected_financials)

    # join selected clickstream + selected financial profile
    df = cs.join(fin, on="Customer_ID", how="left")

    # tidy column order: keys -> financial features -> clickstream features
    df = df.select(["Customer_ID", "snapshot_date"] + selected_financials + selected_clickstream)

    # save gold feature store partition
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())

    return df