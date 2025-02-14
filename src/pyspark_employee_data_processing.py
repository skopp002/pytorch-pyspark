from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType

def analyze_employee_data(employee_df, demographic_df):
    """
    Perform multiple transformations on employee data with demographic information
    Returns a tuple of DataFrames with different analyses
    """
    # Join employee and demographic data
    combined_df = employee_df.join(demographic_df, on="employee_id", how="inner")

    # Stage 1: Calculate department-wise statistics with demographic insights
    dept_stats = combined_df.groupBy("department").agg(
        F.round(F.avg("salary"), 2).alias("avg_salary"),
        F.round(F.stddev("salary"), 2).alias("salary_stddev"),
        F.count("employee_id").alias("employee_count"),
        F.round(F.avg("tenure"), 2).alias("avg_tenure"),
        F.sum("attrition").alias("total_attrition"),
        F.round(F.avg("age"), 2).alias("avg_age"),
        F.round(F.avg("years_of_experience"), 2).alias("avg_experience")
    ).cache()

    # Stage 2: Enhanced salary distribution analysis
    window_spec = Window.orderBy("salary")
    salary_distribution = combined_df.withColumn(
        "salary_percentile", F.ntile(4).over(window_spec)
    ).groupBy("salary_percentile", "department", "education").agg(
        F.count("employee_id").alias("employee_count"),
        F.round(F.avg("salary"), 2).alias("avg_salary"),
        F.round(F.avg("age"), 2).alias("avg_age")
    )

    # Stage 3: Enhanced attrition risk analysis
    risk_analysis = combined_df.withColumn(
        "salary_to_tenure_ratio", F.round(F.col("salary") / F.col("tenure"), 2)
    ).withColumn(
        "is_high_risk", 
        F.when(
            (F.col("salary") < F.lit(75000)) & 
            (F.col("tenure") < F.lit(3)) &
            (F.col("distance_from_office") > F.lit(30)), 
            True
        ).otherwise(False)
    ).groupBy("department", "title", "education").agg(
        F.sum(F.col("is_high_risk").cast("int")).alias("high_risk_count"),
        F.count("employee_id").alias("total_employees"),
        F.round(F.avg("salary_to_tenure_ratio"), 2).alias("avg_salary_tenure_ratio"),
        F.round(F.avg("age"), 2).alias("avg_age")
    )

    # Stage 4: Demographic analysis
    demographic_analysis = combined_df.groupBy("department", "gender", "education").agg(
        F.count("employee_id").alias("employee_count"),
        F.round(F.avg("salary"), 2).alias("avg_salary"),
        F.round(F.avg("age"), 2).alias("avg_age"),
        F.round(F.avg("years_of_experience"), 2).alias("avg_experience"),
        F.sum("attrition").alias("attrition_count")
    )

    return dept_stats, salary_distribution, risk_analysis, demographic_analysis

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
    num_employees = 100

    # Define schema for employee data
    employee_schema = StructType([
        StructField("employee_id", IntegerType(), False),
        StructField("salary", DoubleType(), False),
        StructField("title", StringType(), False),
        StructField("department", StringType(), False),
        StructField("tenure", DoubleType(), False),
        StructField("attrition", IntegerType(), False)
    ])

    # Define schema for demographic data
    demographic_schema = StructType([
        StructField("employee_id", IntegerType(), False),
        StructField("age", IntegerType(), False),
        StructField("gender", StringType(), False),
        StructField("education", StringType(), False),
        StructField("marital_status", StringType(), False),
        StructField("distance_from_office", IntegerType(), False),
        StructField("years_of_experience", IntegerType(), False)
    ])

    # Generate synthetic employee data
    employee_data = {
        'employee_id': range(1, num_employees + 1),
        'salary': np.random.uniform(50000, 150000, num_employees),
        'title': np.random.choice(['Engineer', 'Manager', 'Analyst', 'Director'], num_employees),
        'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], num_employees),
        'tenure': np.random.uniform(1, 15, num_employees),
        'attrition': np.random.choice([0, 1], num_employees, p=[0.7, 0.3])
    }

    # Convert dictionary to list of rows
    employee_rows = [
        (
            employee_data['employee_id'][i],
            float(employee_data['salary'][i]),
            str(employee_data['title'][i]),
            str(employee_data['department'][i]),
            float(employee_data['tenure'][i]),
            int(employee_data['attrition'][i])
        )
        for i in range(num_employees)
    ]

    # Create Spark DataFrame with schema
    employee_df = spark.createDataFrame(employee_rows, schema=employee_schema)

    # Create demographic data
    demographic_data = {
        'employee_id': range(1, num_employees + 1),
        'age': np.random.normal(35, 10, num_employees).astype(int),
        'gender': np.random.choice(['M', 'F', 'Other'], num_employees, p=[0.48, 0.48, 0.04]),
        'education': np.random.choice(
            ['Bachelors', 'Masters', 'PhD', 'High School'], 
            num_employees, 
            p=[0.5, 0.3, 0.1, 0.1]
        ),
        'marital_status': np.random.choice(
            ['Single', 'Married', 'Divorced'], 
            num_employees, 
            p=[0.4, 0.5, 0.1]
        ),
        'distance_from_office': np.random.normal(20, 10, num_employees).astype(int),
        'years_of_experience': np.random.normal(8, 5, num_employees).astype(int)
    }

    # Convert demographic dictionary to list of rows
    demographic_rows = [
        (
            demographic_data['employee_id'][i],
            int(demographic_data['age'][i]),
            str(demographic_data['gender'][i]),
            str(demographic_data['education'][i]),
            str(demographic_data['marital_status'][i]),
            int(demographic_data['distance_from_office'][i]),
            int(demographic_data['years_of_experience'][i])
        )
        for i in range(num_employees)
    ]

    # Create Spark DataFrame for demographic data with schema
    demographic_df = spark.createDataFrame(demographic_rows, schema=demographic_schema)



    # Perform analysis
    dept_stats, salary_distribution, risk_analysis, demographic_analysis = analyze_employee_data(employee_df, demographic_df )
    dept_stats.show()

    # print("\nSalary Distribution by Department and Percentile:")
    salary_distribution.show()

    # print("\nAttrition Risk Analysis:")
    risk_analysis.show()
    
    #print ("Demographic Analysis")
    demographic_analysis.show()
    time.sleep(10000)
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
