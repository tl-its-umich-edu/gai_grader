from google.cloud import bigquery
import os
from dotenv import load_dotenv
import pandas as pd
from canvasapi import Canvas
import requests

from simpleHelper import *
from tqdm import tqdm



 #Load environment file for secrets.
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()

# define function to execute query
def execute_query(sql_file_name, courseId, csv_file_name):
# Initialize a BigQuery client
    client = bigquery.Client()

    # Read in query from text file
    with open(sql_file_name, 'r') as file:
        query = file.read()

    # Execute the query
    query_job = client.query(query)

    # Process the results
    results = query_job.result()
    
    # Convert results to dataframe
    df = pd.DataFrame(data=[list(row.values()) for row in results],
                      columns=list(results.schema))
    # update df columns from ShemaField type to SchemaField.name
    df.columns = [str(col.name) for col in df.columns]

    # write dataframe to csv file
    df.to_csv('data/' + csv_file_name, index=False)


def downloadSubmissionFiles(courseId, assigmentId):
    assignmentSubmissionFolder = os.path.join('data', 'submissions_' + str(assigmentId))

    if not os.path.exists(assignmentSubmissionFolder):
        os.mkdir(assignmentSubmissionFolder)

    API_URL = os.getenv('API_URL')
    API_KEY = os.getenv('API_KEY')

    # Initialize a new Canvas object
    canvas = Canvas(API_URL, API_KEY)

    # Get the course
    course = canvas.get_course(courseId)

    # Get the assignment
    assignment = course.get_assignment(assignmentId)


    # Get the submissions for the assignment
    submissions = assignment.get_submissions()

    # Iterate through the submissions and download the files
    for submission in submissions:
        attachments = submission.attachments
        if attachments:
            for file in attachments:
                file_url = file.url
                file_name = file.filename
                user_id = submission.user_id
                user = course.get_user(user_id)
                user_login_id = user.login_id
                user_sis_id = user.sis_user_id
                # b.name.replace(" ", "_")
                saved_file_name = f"{user_login_id}_{user_id}_{user_sis_id}_{file_name}"
                saved_file_path = os.path.join(assignmentSubmissionFolder, saved_file_name)
                # Download the file based on the file_url, using http get
                download_file(file_url, saved_file_path)

# write a python function to download file based on url
def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

def step_1_data_prep(courseId, assignmentId, courseName):
    # Step 1: Execute queries and save results to csv files

    execute_query('query_assignments.sql', courseId, f'{courseName}_{courseId}_assignments.csv')
    execute_query('query_rubrics.sql', courseId, f'{courseName}_{courseId}_rubrics.csv')
    execute_query('query_submissions.sql', courseId, f'{courseName}_{courseId}_gradings.csv')
    execute_query('query_submission_comments.sql', courseId, f'{courseName}_{courseId}_comments.csv')

    # Step 2: Generate criterion file
    pathToAssgnCSV = os.path.join('data',f'{courseName}_{courseId}_assignments.csv')
    pathToGradingsCSV = os.path.join('data',f'{courseName}_{courseId}_gradings.csv')
    pathToRubricsCSV = os.path.join('data',f'{courseName}_{courseId}_rubrics.csv')
    pathToCriterionCSV = os.path.join('data',f'{courseName}_{courseId}_criterion.csv')

    makeBlankCriterionFile(pathToAssgnCSV, pathToGradingsCSV, pathToRubricsCSV, pathToCriterionCSV)
    

    '''
    # Step 3: 
    Generate text versions of submission files
    You will need to run this notebook cell to convert the submissions into a format usable by the notebook.
    The data/submissions_xxxxxx should contain the unzipped submission folders that are exported from Canvas. The folder name can be anything, you just need to specify it in the originalSubmissionFolder variable. You will also need the Assignment ID in variable assignmentID.
    A folder for converted submissions is automatically made called data/Converted submissions_xxxxxx where xxxxxx is the Assignment ID.
    '''

    originalSubmissionFolder = os.path.join('data', f'submissions_{assignmentId}')
    downloadSubmissionFiles(courseId, assignmentId)
    convertSubmissions(originalSubmissionFolder, assignmentId)

'''
Part 2: Getting ChatGPT responses for submissions
You need to provide the full file path to each of the 4 CSV files. You also need to specify your OpenAI key details, the course code, and the assignment ID. The courseName variable simply is the real name of the course.
'''
def step_2_get_chatGPT_responses(courseId, assignmentId, courseName):
    pathToAssgnCSV = os.path.join('data',f'{courseName}_{courseId}_assignments.csv')
    pathToGradingsCSV = os.path.join('data',f'{courseName}_{courseId}_gradings.csv')
    pathToRubricsCSV = os.path.join('data',f'{courseName}_{courseId}_rubrics.csv')
    pathToCriterionCSV = os.path.join('data',f'{courseName}_{courseId}_criterion.csv')

    overWriteSave = False

    customDescMode = True
    critDescDF = pd.read_csv(pathToCriterionCSV).drop_duplicates()

    saveFolder = 'data/saves'
    errorFolder = 'data/error'
    if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
    if not os.path.exists(errorFolder):
            os.mkdir(errorFolder)
        
    gradeRubricAssignmentDF = getGRAData(pathToAssgnCSV, pathToGradingsCSV, pathToRubricsCSV)
    print(gradeRubricAssignmentDF)
    print(f"gradeRubricAssignmentDF size {gradeRubricAssignmentDF.shape[0]}")
    print(assignmentId)

    gradeRubricAssignmentDF = gradeRubricAssignmentDF[gradeRubricAssignmentDF['assignment_id']==assignmentId]

    print(f"gradeRubricAssignmentDF size {gradeRubricAssignmentDF.shape[0]}")

    for index, row in (pbar := tqdm(gradeRubricAssignmentDF.iterrows(), total=gradeRubricAssignmentDF.shape[0])):
        pbar.set_description(f"Processing: {assignmentId}-{row['submitter_id']}")
        if checkIfSaved(row['assignment_id'], row['submitter_id'], saveFolder, errorFolder) and not overWriteSave:
            # print(f"Already saved: {assignmentID}-{row['submitter_id']}")
            continue
        else:
            dataDict, runSuccess = processGRARow(row, courseName, customDescMode, critDescDF)
            saveOutputasPickle(dataDict, runSuccess, saveFolder, errorFolder)

    print(f"gradeRubricAssignmentDF size {gradeRubricAssignmentDF.shape[0]}")

'''
Part 3: Analyzing the results and making charts and tables
'''
def step_3_analyze_results(courseId, assignmentId, courseName):

    pathToAssgnCSV = os.path.join('data',f'{courseName}_{courseId}_assignments.csv')
    pathToGradingsCSV = os.path.join('data',f'{courseName}_{courseId}_gradings.csv')
    pathToRubricsCSV = os.path.join('data',f'{courseName}_{courseId}_rubrics.csv')
    pathToCriterionCSV = os.path.join('data',f'{courseName}_{courseId}_criterion.csv')

    saveFolder = 'data/saves'
    errorFolder = 'data/error'
    if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
    if not os.path.exists(errorFolder):
            os.mkdir(errorFolder)

    resultsDF = convertPicklesToDF(saveFolder)
    errorDF = convertPicklesToDF(errorFolder)

    excelFolder = 'data/excelJK'
    if not os.path.exists(excelFolder):
            os.mkdir(excelFolder)
    chartFolder = 'data/chartsJK'
    if not os.path.exists(chartFolder):
            os.mkdir(chartFolder)

    saveName = f'{courseName}-{assignmentId}'
            
    mergedCriterionData = getCriterionDataDF(resultsDF, saveName, excelFolder)

    meanDiffDF, meanDiffPercentDF = saveGraderPeerGPTMeanScoreDiff(resultsDF, saveName, excelFolder)
    print(meanDiffPercentDF)

    print("sss")
    critDataDF = pd.DataFrame()
    for index,row in resultsDF.iterrows():
        criterionData = row['data_peerGPT']
        for col in ['submitter_id', 'assignment_id', 'grader_id']:
            criterionData[col] = row[col]
        critDataDF = pd.concat([critDataDF, criterionData])
    allCritDF = critDataDF.drop(['mastery_points','ignore_for_scoring','title','peerGPT_criterion_id','description_grade'],
                            axis=1, errors='ignore')

    print(f"allCritDF size {allCritDF.shape[0]}")

    meanInfoList = []
    for group in allCritDF.groupby(['assignment_id','criterion_id','grader_id']):
        meanInfoList.append({'assignment_id':group[0][0], 'criterion_id':group[0][1], 'grader_id':group[0][2], \
                            'Grader Mean':group[1]['points_grade'].mean(), \
                            'Grader Std. Dev.':group[1]['points_grade'].std(), \
                            'peerGPT Mean':group[1]['peerGPT_criterion_score'].mean(), \
                            'peerGPT Std. Dev.':group[1]['peerGPT_criterion_score'].std(), \
                            # 'Correlation Score':group[1]['peerGPT_criterion_score'].corr(group[1]['points_grade']), \
                            })
    meanInfoDF = pd.DataFrame(meanInfoList)
    meanInfoDF['Mean Difference'] = meanInfoDF['peerGPT Mean'] - meanInfoDF['Grader Mean']
    gradeRubricAssignmentDF = getGRAData(pathToAssgnCSV, pathToGradingsCSV, pathToRubricsCSV)
    print(f"gradeRubricAssignmentDF size {gradeRubricAssignmentDF.shape[0]}")
    assignmentDF = gradeRubricAssignmentDF[['assignment_id', 'assignment_title']].drop_duplicates()
    print(f"assignmentDF size {assignmentDF.shape[0]}")

    baseInfoDF = allCritDF[['assignment_id', 'criterion_id', 'description_rubric', 'points_rubric']].drop_duplicates()
    baseInfoDF = baseInfoDF.merge(assignmentDF, on='assignment_id')
    print(f"baseInfoDF size {baseInfoDF.shape[0]}")

    globalMeanList = [{'assignment_id':group[0][0], 'criterion_id':group[0][1], \
                    'All Graders Mean':group[1]['points_grade'].mean(), \
                    'All Graders Std. Dev.':group[1]['points_grade'].std(), \
                    'Global peerGPT Mean':group[1]['peerGPT_criterion_score'].mean(), \
                    'Global peerGPT Std. Dev.':group[1]['peerGPT_criterion_score'].std()} \
                        for group in allCritDF.groupby(['assignment_id', 'criterion_id'])]
    globalMeanDF = pd.DataFrame(globalMeanList)

    baseInfoDF = baseInfoDF.merge(globalMeanDF, on=['assignment_id', 'criterion_id'])

    fullInfoDF = meanInfoDF.merge(baseInfoDF, on=['assignment_id', 'criterion_id'])
    fullInfoDF['Mean Difference %'] = 100*fullInfoDF['Mean Difference'].div(fullInfoDF['points_rubric'])
    print(fullInfoDF)
    fullInfoDF['Grader Mean Diff. %'] = 100*(fullInfoDF['All Graders Mean'] - fullInfoDF['Grader Mean']).div(fullInfoDF['points_rubric'])
    print(fullInfoDF)
    print(f"sSaving file at: {os.path.join(excelFolder, saveName+' - Grader Difference Table.xlsx')}")
    fullInfoDF.to_excel(os.path.join(excelFolder, saveName+' - Grader Difference Table.xlsx'))

    rubricInfo = gradeRubricAssignmentDF[['assignment_id', 'data_rubric']].drop_duplicates('assignment_id').reset_index(drop=True)
    rubricOrderDict = {}
    for index, row in rubricInfo.iterrows():
        rubricOrderDict[row['assignment_id']] = pd.DataFrame(row['data_rubric'])['description'].tolist()

    getZScoreAndCI(fullInfoDF, saveName, excelFolder, confidence=0.93)

    getMeanDiffPercentCharts(fullInfoDF, rubricOrderDict, courseName, chartFolder)



courseId = 656488
assignmentId = 2217018
courseName = 'MovementScience'

# Step 1: Execute queries and save results to csv files
step_1_data_prep(courseId, assignmentId, courseName)

# Step 2: Get ChatGPT responses for submissions
step_2_get_chatGPT_responses(courseId, assignmentId, courseName)

# Step 3: Analyze the results and make charts and tables
step_3_analyze_results(courseId, assignmentId, courseName)

