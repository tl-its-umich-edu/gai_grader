# GraderGPT - - README.md

Making GenAI help humans grade better.

## Breakdown of the main files present in this project:
1. Library function files:
   > `helper.py` : Contains all functions needed to retrieve the CSV and submission data, build a prompt, and get gradings from ChatGPT.  
                   Contains the config class used to set parameters such as version names and runtime parameters.  
   > `postProcessHelper.py` : Contains functions primarily focused with the analysis and viewing of the data after getting grades from ChatGPT.  

2. Grading scripts:
   > `GraderGPT_SP.ipynb` : Notebook : Used to get gradings for a course, running one submission at a time. Will not overload requests made to the API.  
   > `GraderGPT_MP.py` : Script : Same as above, but can run in multiprocess, and hence run much faster. Be careful about the number of requests made.  

3. Submission converter notebook:
   > `GraderGPT_MiscFunctions.ipynb` : Primary use : Convert Canvas exported (unzipped) submission folders of .pdf and .docx submissions into a directory of .txt submissions.  
                                 Secondary uses : Check token counts of requests to be made and identify if any might go above the token limit.  
                                 Secondary uses : Save a fully constructed and filled out prompt to file.  
                                 Secondary uses : Build and save a `<FULL_COURSE_NAME>_criterions.csv` file for using custom rubric descriptions.  

4. Post-processing notebook:
   > `GraderGPT_postProcess.ipynb` : Contains various function calls to generate charts and tables pertinent to analysis of the gradings from ChatGPT.

## Key variables and config parameters: 
Refer the `.env.sample` file to build the required `.env` file to run the scripts and more information on key variables used there as well.  
Some variables can be overridden from the `.env` file while running the notebooks and scripts. See `GraderGPT_SP.ipynb` for example of using both options.  
`GraderGPT_postProcess.ipynb` uses custom overrides to not require manual editing of the `.env` file and restarting during analysis.  

## Basic Setup Instructions:
1. Get OpenAI key for accessing API:
(This is only if you are using a personal account. The code is not configured for this setup currently.)
   > Using OpenAI's Web UI, click on "Profile Picture" (Top right of screen) > "View API keys" (https://platform.openai.com/account/api-keys)  
   > Make a new key, and copy the key for use in accessing the API.  
(If using the Azure UM endpoint, please ref ____ for steps)

2. Use `requirements.txt` to install the needed Python libraries for running all the scripts.
Run the command: 
   > pip install -r requirements.txt

3. Use the `.env.sample` in the config folder to create an `.env` file in the root of the directory. 
   > Specify the API key for OpenAI using the key you copied in step 2.
   <!-- > Specify the settings for DB key using values from the `database-research_ro.json` file. -->
Note: You need to have these keys, there are no default values.

4. Adjust variables in the `.env` file before running if not applying defaults.

5. You need to provide a set of different files for the code to work in the folder 'data', primarily being the 'CSV Data' and 'Submissions'.  
   a. CSV Data:  
   > You need to provide 3 files from Canvas, containing assignment, grading, and rubric data.  
   > Each file name has a prefix that Canvas generates, that follows a format: `(name)\_(3/4 digit code)\_(2 letters for term)\_(year)\_(courseID)\_`  
      (Note: MWrite courses will also have a postfix `MWrite\_` at the end of the prefix.)  
      The file names must follow the naming schema:  
         For assignment data: `<FULL_COURSE_NAME>_assignments.csv`  
         For grading data: `<FULL_COURSE_NAME>_gradings.csv`  
         For rubric data: `<FULL_COURSE_NAME>_rubrics.csv`  

   > If you are using custom criterion data such as grader notes, you also need a criterion data file named `<FULL_COURSE_NAME>_criterions.csv`.  
      Use `GraderGPT_MiscFunctions.ipynb` to build the custom criterions file based on the course you are working with.
      > Here, for each criteria listed, you must add the supplemental grading notes for that criteria in the `custom_description` column.  
   
   > If you are using custom assigment decription data extracted from the `assignment_description` column in the `<FULL_COURSE_NAME>_assignments.csv`:  
   You will need to add a `cleaned_description` column that contains this custom data.  
   > If you have assignments that use submission data from a different assignment, you must specify the source `assignment_id` in the `<FULL_COURSE_NAME>_assignments.csv`:  
   You will need to add a `file_submission_source` column that contains this additional data.  
   If using this column to specify the source `assignment_id`, any blank cells in the column will cause that assignment to be skipped for grading.  
   Hence, for assignments whose submissions are present within the same assignment, the `file_submission_source` will simply be its own `assignment_id`.

   b. Submissions:  
   > You need to provide the submission data as folders per assignment, where the assignment ID is the folder name.  
   The file names generated by Canvas should not be edited. .docx and .pdf file formats are supported, but .docx is preferred.  
   If running for the first time, use the `GraderGPT_MiscFunctions.ipynb` notebook to generate Text File versions of the submission data.  
