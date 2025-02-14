from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time


def analyze_employee_data(df):
    """
    Perform multiple transformations on employee data
    Returns a tuple of DataFrames with different analyses
    """
    # Stage 1: Calculate department-wise statistics
    dept_stats = df.groupBy("department").agg(
        F.round(F.avg("salary"), 2).alias("avg_salary"),
        F.round(F.stddev("salary"), 2).alias("salary_stddev"),
        F.count("employee_id").alias("employee_count"),
        F.round(F.avg("tenure"), 2).alias("avg_tenure"),
        F.sum("attrition").alias("total_attrition")
    ).cache()  # Cache this result as it will be used multiple times

    # Stage 2: Calculate salary bands and employee distribution
    window_spec = Window.orderBy("salary")
    salary_distribution = df.withColumn(
        "salary_percentile", F.ntile(4).over(window_spec)
    ).groupBy("salary_percentile", "department").agg(
        F.count("employee_id").alias("employee_count"),
        F.round(F.avg("salary"), 2).alias("avg_salary")
    )

    # Stage 3: Analyze attrition risk factors
    risk_analysis = df.withColumn(
        "salary_to_tenure_ratio", F.round(F.col("salary") / F.col("tenure"), 2)
    ).withColumn(
        "is_high_risk", 
        F.when(
            (F.col("salary") < F.lit(75000)) & 
            (F.col("tenure") < F.lit(3)), 
            True
        ).otherwise(False)
    ).groupBy("department", "title").agg(
        F.sum(F.col("is_high_risk").cast("int")).alias("high_risk_count"),
        F.count("employee_id").alias("total_employees"),
        F.round(F.avg("salary_to_tenure_ratio"), 2).alias("avg_salary_tenure_ratio")
    )

    return dept_stats, salary_distribution, risk_analysis

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Employee Data Analysis") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
        .master("local[*]") \
        .getOrCreate()

    # Generate synthetic data
    synthetic_data = {
        'employee_id': range(1, 101),  # Increased to 100 employees
        'salary': np.random.uniform(50000, 150000, 100),
        'title': np.random.choice(['Engineer', 'Manager', 'Analyst', 'Director'], 100),
        'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], 100),
        'tenure': np.random.uniform(1, 15, 100),
        'attrition': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    }

    # Convert to Spark DataFrame
    pandas_df = pd.DataFrame(synthetic_data)
    df = spark.createDataFrame(pandas_df)

    # Perform analysis
    dept_stats, salary_distribution, risk_analysis = analyze_employee_data(df)
    dept_stats.show()

    # print("\nSalary Distribution by Department and Percentile:")
    salary_distribution.show()

    # print("\nAttrition Risk Analysis:")
    risk_analysis.show()
    time.sleep(10000)
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
