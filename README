This project demonstrates the similarities between 2 popular frameworks - PySpark and PyTorch from seemingily independent realms, compares the concepts and significance

Below steps are to enable you to try the jobs and use the interactive visualization aspects yourself. 
You can skip them if you would like to understand the concepts and would like to view the results through screenshots

Understanding the Visuals:

Spark UI ![Spark DAG Visualization](screenshots/pyspark_dag_with_join.png) reveals the execution plan. The nodes in the DAG represent intermediate computation steps. 
For instance, in this diagram, we have WholeStageCodegen(1) which represents RDD built out of dataset 1 (employee_data) and WholeStageCodegen(1) represents RDD built with 
the 2nd dataset (demographic_data). Each stage in spark shows the progression of computation on these datasets. 

Constrasting this to the DAG in PyTorch [TensorBoardVisuals](screenshots/TensorBoardVisualizingDAG-pytorch.png), we see a similar structure representing layers of the Neural Network model

Steps to execute the PySpark and PyTorch jobs in order to visualize and explore the WebUI interactively:
1. python -m pip venv .venv
2. pip install -r requirements.txt
3. In Terminal 1 python pyspark_employee_data_processing.py. This code has sleep added to enable users to explore the interactive Spark UI. The UI will be active at http://localhost:4040/jobs/
4. In Terminal 2 python pytorch_dag_illustration.py --> creates a file "hr_attrition_model.pth" in current working directory
5. Launch Tensorboard to visualize the model - 
tensorboard --logdir="./" 
This will launch Tensorboard at http://localhost:6006/
You can optionally use  torch-tb-profiler which allows for Visual Studion built in tensorboard




Troubleshooting:
If you run into issues with execution of the pyspark job, follow below steps:
For Spark, had to install Java:
 1. echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> /Users/skoppar/.zshrc
  
  For compilers to find openjdk you may need to set:
  export CPPFLAGS="-I/opt/homebrew/opt/openjdk/include"
 2. source ~/.zshrc
