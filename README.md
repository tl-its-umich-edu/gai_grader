# README.md

Making GenAI help humans grade better. This is a simplified self-contained version of the code. 
It is mean to grade a single assignment in a single course at a time. 

## Breakdown of the main files present in this project:
1. Library function files:
   > `simpleHelper.py` : Contains all functions needed to retrieve the CSV and submission data, build a prompt, and get gradings from ChatGPT.  
                         Contains functions primarily focused with the analysis and viewing of the data after getting grades from ChatGPT.  

2. Main Notebook:
   > `GraderGPT_SimpleNotebook.ipynb` : Notebook : Consists of 3 parts, Pre-processing, Grading, and Analysis. Runs in single processing only.  



Use `requirements.txt` to install the needed Python libraries for running the notebook.
Run the command: 
   > pip install -r requirements.txt

Based on your operating system, you also need to install pandoc, following instructions from here: https://pandoc.org/installing.html


If you are using custom assigment decription data extracted from the `assignment_description` column in the `xxxx_assignments.csv`:  
   You will need to add a `cleaned_description` column that contains this custom data.  
If you have assignments that use submission data from a different assignment, you must specify the source `assignment_id` in the `xxxx_assignments.csv`:  
   You will need to add a `file_submission_source` column that contains this additional data.  
If using this column to specify the source `assignment_id`, any blank cells in the column will cause that assignment to be skipped for grading.  
   Hence, for assignments whose submissions are present within the same assignment, the `file_submission_source` will simply be its own `assignment_id`.