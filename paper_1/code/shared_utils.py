"""
Shared Utilities for Network Intrusion Detection System (NIDS).
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType

# -- Python & Java runtime configuration --
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[4] pyspark-shell'


# SPARK SESSION INITIALIZATION

def create_spark_session(app_name: str = "IDS_Binary_Prediction") -> SparkSession:
    """Create a SparkSession optimised for local-mode processing."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "8g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "100s")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# DATA PREPROCESSING FUNCTIONS

def clean_column_names(df):
    """Normalise column names to lowercase snake_case."""
    for col_name in df.columns:
        new_name = (
            col_name.strip()
            .lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        while "__" in new_name:
            new_name = new_name.replace("__", "_")
        df = df.withColumnRenamed(col_name, new_name)
    return df


def handle_infinity_values(df):
    """Replace Infinity and NaN values with null."""
    for col_name in df.columns:
        if dict(df.dtypes)[col_name] in ["double", "float"]:
            df = df.withColumn(
                col_name,
                F.when(
                    (F.col(col_name).isNull())
                    | (F.isnan(F.col(col_name)))
                    | (F.col(col_name) == float("inf"))
                    | (F.col(col_name) == float("-inf")),
                    None,
                ).otherwise(F.col(col_name)),
            )
    return df


def align_schema(df, ref_columns: list):
    """Align DataFrame schema by adding missing columns as null."""
    for c in ref_columns:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(StringType()))
    return df.select(ref_columns)
