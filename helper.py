import logging
import os
import sys
from numpy.random import default_rng
import time
import pickle
import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv()

import openai

logging.basicConfig(
    format='%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    level=logging.INFO
)

def getGRAData(config, mpMode=False):
    gradeCSVFile = f'{config.courseName}gradings.csv'
    rubricCSVFile = f'{config.courseName}rubrics.csv'
    assignmentCSVFile = f'{config.courseName}assignments.csv'
    
    gradeDataDF = pd.read_csv(os.path.join(config.CSVDataFolder,gradeCSVFile))
    rubricDataDF = pd.read_csv(os.path.join(config.CSVDataFolder,rubricCSVFile))
    assignmentDataDF = pd.read_csv(os.path.join(config.CSVDataFolder,assignmentCSVFile))

    gradeDataDF['data'] = gradeDataDF['data'].apply(lambda dataJSON: json.loads(dataJSON))
    rubricDataDF['data'] = rubricDataDF['data'].apply(lambda dataJSON: json.loads(dataJSON))

    gradeRubricDF = gradeDataDF.merge(rubricDataDF, on='rubric_id', suffixes=('_grade', '_rubric'))
    gradeRubricAssignmentDF = gradeRubricDF.merge(assignmentDataDF, on='assignment_id', 
                                                  suffixes=('', '_assignment'))

    gradeRubricAssignmentDF = gradeRubricAssignmentDF[['submitter_id', 'grader_id', 'score', 'rubric_id', 
                                                       'assignment_id', 'assignment_title', 'data_grade', 
                                                       'data_rubric', 'points_possible', 
                                                       'assignment_description', 'cleaned_description']]
    if mpMode:
        rowDataList = []
        for index, row in gradeRubricAssignmentDF.iterrows():
            rowDataList.append({'row':row, 'config':config})
        return rowDataList
    else:    
        return gradeRubricAssignmentDF


def promptBuilder(promptVariablesDict=None, saveTemplate=False, config=None):
    if saveTemplate:
        promptVariablesDict = {promptVariable:f'<<ENTER {promptVariable.upper()} HERE>>' for promptVariable in config.promptVariableNames}

    starterText = f'''You are a grader for the course "{promptVariablesDict['Course Name']}". 
Your task is to grade a student's submission for the assignment "{promptVariablesDict['Assignment Name']}" using the provided criteria in the context of this course. 
You will follow these specific rubric criteria to assign points related to different aspects of the assignment.
The assignment's summary is "{promptVariablesDict['Assignment Description']}". 
Each criteria has guidelines used to grade that will inform you on how to make penalties and leave feedback. Use the guidelines per criteria to assign a criteria score and feedback.
The points assigned must lie between 0 and the max points as listed for each criteria.
The student's submission is delimited by triple backticks.
The criteria are:
'''

    endText = f'''The student submission is:
```{promptVariablesDict['Student Submission']}```
For each criterion listed, return the assigned points, and feedback comment of under 100 words based on the criterion guidelines and errors made.
Use the format:
<criterion 1 ID> : <criterion 1 score> : <comment>
<criterion 2 ID> : <criterion 2 score> : <comment>
.
.
.
'''
    fullText = starterText + promptVariablesDict['Criterion Description and Rubric'] + endText
    return fullText

def buildCritPrompt(criterionDF, useCustomDesc=True):
    fullCritText = ''
    for index, cRow in criterionDF.iterrows():
        if useCustomDesc:
            criteriaText = f"{index+1}. Criterion Title: '{cRow['description_rubric']}', \
CriterionID: '{cRow['criterion_id'] }', \
Max Points: '{cRow['points_rubric'] }', \
\nCriterion Guidelines: {cRow['custom_description']}\n"
        else:
            ratingsTextList = [f'\t{rating["description"]} : {rating["points"]} points\n' 
                           for rating in cRow['ratings']]
            if cRow['long_description']:
                criteriaText = f"{index+1}. Criterion Title: '{cRow['description_rubric']}', CriterionID: '{cRow['criterion_id'] }', \
                    \nCriterion Description: '{cRow['long_description']}', \
                    \nRatings Guide:\n"+''.join(ratingsTextList)
            else:
                criteriaText = f"{index+1}. Criterion Title: '{cRow['description_rubric']}', CriterionID: '{cRow['criterion_id'] }', \
                        \nRatings Guide:\n"+''.join(ratingsTextList)
        fullCritText += criteriaText
    return fullCritText

def processResponse(responseText):
    critScores = []
    responseLines = responseText.split('\n')
    for line in responseLines:
        if ':' in line:
            if len(line.split(':'))==3:
                id, score, reason = line.split(':')
                score = float(score.strip().split()[0].strip())
                critScores.append({'peerGPT_criterion_id':id.strip(), 'peerGPT_criterion_score':score, 'peerGPT_reason':reason.strip()})
        else:
            critScores[-1]['peerGPT_reason'] += line
    return pd.DataFrame(critScores)

def getSubmissionText(assignmentID, userID, textSubmissionsFolder='Text Submissions'):
    submissionFilePath = os.path.join(textSubmissionsFolder, f'{assignmentID}', f'{userID}.txt')
    if os.path.exists(submissionFilePath):
        with open(submissionFilePath) as textFile:
            submissionLines = textFile.readlines()
        studentSubmission = ''.join([line.strip() if line!='\n' else '\n' for line in submissionLines])
        studentSubmission = studentSubmission.replace('\n\n', '\n')
        return studentSubmission
    else:
        return False

def checkIfSaved(assignmentID, userID, config):
    saveName = f"{assignmentID}-{userID}.p"
    for subFolder in config.saveFolders:
        fileSavePath = os.path.join(config.saveFolderPath[subFolder], saveName)
        if os.path.exists(fileSavePath):
            return True
    return False

def processTokenCount(row, config):        
    gradeDataDF = pd.DataFrame(row['data_grade'])
    rubricDataDF = pd.DataFrame(row['data_rubric'])
    descDataDF = config.critDescDF[config.critDescDF['assignment_id']==row['assignment_id']][['custom_description', 'id']]

    fullCriterionDF = gradeDataDF.merge(rubricDataDF, left_on='criterion_id', 
                                        right_on='id', suffixes=('_grade', '_rubric'))
    fullCriterionDF = fullCriterionDF.merge(descDataDF, left_on='criterion_id', right_on='id')
    fullCriterionDF = fullCriterionDF.drop(['id_grade', 'id_rubric', 'learning_outcome_id', 'id',
                                            'comments_enabled', 'comments_html', 'criterion_use_range'], 
                                            axis=1, errors='ignore')    
    # print(row)
    # display(fullCriterionDF)

    fullCritText = buildCritPrompt(fullCriterionDF, True)
    studentSubmission = getSubmissionText(row['assignment_id'], row['submitter_id'])

    if studentSubmission:
        promptVariableDict = {
                            'Course Name': config.simpleCourseName,
                            'Assignment Name': row['assignment_title'],
                            'Assignment Description': row['cleaned_description'],
                            'Student Submission': studentSubmission,
                            'Criterion Description and Rubric': fullCritText,
                            'Maximum Points': row['points_possible'],
                            }
        fullPrompt = promptBuilder(promptVariableDict)
        # print(fullPrompt)
        return fullPrompt
    else:
        return False

def processGRARow(row, config):        
    gradeDataDF = pd.DataFrame(row['data_grade'])
    rubricDataDF = pd.DataFrame(row['data_rubric'])
    descDataDF = config.critDescDF[config.critDescDF['assignment_id']==row['assignment_id']][['custom_description', 'id']]

    fullCriterionDF = gradeDataDF.merge(rubricDataDF, left_on='criterion_id', 
                                        right_on='id', suffixes=('_grade', '_rubric'))
    fullCriterionDF = fullCriterionDF.merge(descDataDF, left_on='criterion_id', right_on='id')
    fullCriterionDF = fullCriterionDF.drop(['id_grade', 'id_rubric', 'learning_outcome_id', 'id',
                                            'comments_enabled', 'comments_html', 'criterion_use_range'], 
                                            axis=1, errors='ignore') 
    # print(row)
    # display(fullCriterionDF)

    fullCritText = buildCritPrompt(fullCriterionDF, True)
    studentSubmission = getSubmissionText(row['assignment_id'], row['submitter_id'])

    if studentSubmission:
        promptVariableDict = {
                            'Course Name': config.simpleCourseName,
                            'Assignment Name': row['assignment_title'],
                            'Assignment Description': row['cleaned_description'],
                            'Student Submission': studentSubmission,
                            'Criterion Description and Rubric': fullCritText,
                            'Maximum Points': row['points_possible'],
                            }
        fullPrompt = promptBuilder(promptVariableDict)
        # print(fullPrompt)
        # with open(os.path.join(config.versionOutputFolder, f'{config.fullName}_exampleFilledPrompt.txt'), 'w') as textFile:
        #     textFile.write(fullPrompt)
        # print(x)
       
        peerBot = peerGPT(config)
        response, responseSucess = peerBot.get_completion(fullPrompt)
        del peerBot
        
        if responseSucess:
            scoreBreakdownDF = processResponse(response['Text'])
            finishedCriterionDF = fullCriterionDF.merge(scoreBreakdownDF, 
                                                        left_on='criterion_id', 
                                                        right_on='peerGPT_criterion_id', 
                                                        suffixes=('', '_predicted'))
            finishedCriterionDF = finishedCriterionDF.drop(['long_description', 'ratings'], axis=1)
            # display(finishedCriterionDF)
            savedRowDF = row[['submitter_id', 'grader_id', 'rubric_id', 
                              'assignment_id', 'score', 'points_possible']]
            savedRowDF['data_peerGPT'] = finishedCriterionDF
            savedRowDF['peerGPT_score'] = finishedCriterionDF['peerGPT_criterion_score'].sum()
            savedRowDict = savedRowDF.to_dict()
            # display(savedRowDF)
            return savedRowDict, True
        else:
            return {'assignment_id':row['assignment_id'], 'submitter_id':row['submitter_id'], 
                    'Error':response}, False
    else:
        return {'assignment_id':row['assignment_id'], 'submitter_id':row['submitter_id'], 
                'Error':'Submission File Missing.'}, False
    
def checkRunSave(row, config):
    if checkIfSaved(row['assignment_id'], row['submitter_id'], config) and not config.overWriteSave:
        return False
    else:
        dataDict, runSuccess = processGRARow(row, config)
        saveOutputasPickle(dataDict, runSuccess, config)
        return True
    
def checkRunSaveMP(rowData):
    if checkIfSaved(rowData['row']['assignment_id'], rowData['row']['submitter_id'], rowData['config']) and not rowData['config'].overWriteSave:
        return False
    else:
        time.sleep(default_rng().uniform(1,2))
        dataDict, runSuccess = processGRARow(rowData['row'], rowData['config'])
        saveOutputasPickle(dataDict, runSuccess, rowData['config'])
        return True

def saveOutputasPickle(dataDict, runSuccess, config):
    saveName = f"{dataDict['assignment_id']}-{dataDict['submitter_id']}.p"
    if runSuccess:
        pickleSavePath = os.path.join(config.saveFolderPath['pickle'], saveName)
    else:
        pickleSavePath = os.path.join(config.saveFolderPath['error'], saveName)

    with open(pickleSavePath, 'wb') as handle:
        pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def convertPicklesToDF(folderName, config):
    saveDataList = []
    for savedPickle in os.listdir(config.saveFolderPath[folderName]):
        rowSaveData = pickle.load(open(os.path.join(config.saveFolderPath[folderName], savedPickle), 'rb'))
        # if folderName=='pickle':
        #     if 'peerGPT_criterion_score' in rowSaveData['data_peerGPT']:
        #         rowSaveData['peerGPT_score_real'] = rowSaveData['data_peerGPT']['peerGPT_criterion_score'].sum()
        #     else:
        #         rowSaveData['peerGPT_score_real'] = rowSaveData['data_peerGPT']['peerGPT__criterion_score'].sum()
        saveDataList.append(rowSaveData)
    resultsDF = pd.DataFrame(saveDataList)
    return resultsDF

class Config:
    def __init__(self):
        self.logLevel = logging.INFO

        self.openAIParams: dict = {'KEY': None,
                                   'BASE':None,
                                   'VERSION':None,
                                   'TYPE':None,
                                   'DEPLOYMENT_NAME':None}
        
        self.rootSubmissionFolder: str = 'Submissions'
        self.CSVDataFolder: str ='CSV Data'
        self.ExcelDumpFolder: str = 'Excel Dumps'
        self.ChartDumpFolder: str = 'Chart Dumps'
        self.saveFolders = {
                            'pickle':'pickledSaves',
                            'error':'errorData'
                            }
        self.saveFolderPath = {}
        self.courseName: str = 'MOVESCI_110_WN_2023_584988_MWrite_'
        self.simpleCourseName: str = 'Movement Science'

        self.promptVariableNames = [
                    'Course Name',
                    'Assignment Name',
                    'Assignment Description',
                    'Student Submission',
                    'Criterion Description and Rubric',
                    'Maximum Points',
                    ]

        self.overWriteSave: bool = False
        self.fullName: str = None
        self.versionOutputFolder = None
        self.poolSize: int = 8

        self.critDescDF = pd.read_excel(os.path.join(self.CSVDataFolder,'Criterion Info Sheet.xlsx'))
        del self.critDescDF['Unnamed: 0']

        self.textSubmissionsFolder: str = 'Text Submissions'
        if not os.path.exists(self.textSubmissionsFolder):
            os.mkdir(self.textSubmissionsFolder)
        self.tempFolder: str = 'temp'
        if not os.path.exists(self.tempFolder):
            os.mkdir(self.tempFolder)
        self.rootOutputFolder: str = 'Output'
        if not os.path.exists(self.rootOutputFolder):
            os.mkdir(self.rootOutputFolder)

    def setSaveDetails(self, fullName):
        self.fullName = fullName
        self.versionOutputFolder = os.path.join(self.rootOutputFolder,self.fullName)
        for subFolder in self.saveFolders:
            if not os.path.exists(os.path.join(self.rootOutputFolder,self.fullName,subFolder)):
                os.makedirs(os.path.join(self.rootOutputFolder,self.fullName,subFolder))
            self.saveFolderPath[subFolder] = os.path.join(self.rootOutputFolder,self.fullName,subFolder)
        return True

    def set(self, name, value):
        if name in self.__dict__:
            self.name = value
        else:
            raise NameError('Name not accepted in set() method')

    def configFetch(self, name, default=None, casting=None, validation=None, valErrorMsg=None):
        value = os.environ.get(name, default)
        if (casting is not None):
            try:
                value = casting(value)
            except ValueError:
                errorMsg = f'Casting error for config item "{name}" value "{value}".'
                logging.error(errorMsg)
                return None

        if (validation is not None and not validation(value)):
            errorMsg = f'Validation error for config item "{name}" value "{value}".'
            logging.error(errorMsg)
            return None
        return value

    def setFromEnv(self):
        try:
            self.logLevel = str(os.environ.get(
                'LOG_LEVEL', self.logLevel)).upper()
        except ValueError:
            warnMsg = f'Casting error for config item LOG_LEVEL value. Defaulting to {logging.getLevelName(logging.root.level)}.'
            logging.warning(warnMsg)

        try:
            logging.getLogger().setLevel(logging.getLevelName(self.logLevel))
        except ValueError:
            warnMsg = f'Validation error for config item LOG_LEVEL value. Defaulting to {logging.getLevelName(logging.root.level)}.'
            logging.warning(warnMsg)

        # Currently the code will check and validate all config variables before stopping.
        # Reduces the number of runs needed to validate the config variables.
        envImportSuccess = True

        for credPart in self.openAIParams:
            self.openAIParams[credPart] = self.configFetch(
                'OPENAI_API_' + credPart, self.openAIParams[credPart], str)
            envImportSuccess = False if not self.openAIParams[credPart] or not envImportSuccess else True

        if not envImportSuccess:
            sys.exit('Exiting due to configuration parameter import problems.')
        else:
            logging.info('All configuration parameters set up successfully.')

    def saveTemplatePrompt(self):
        with open(os.path.join(self.versionOutputFolder, f'{self.fullName}_templatePrompt.txt'), 'w') as textFile:
            textFile.write(promptBuilder(saveTemplate=True, config=self))
        return True


class peerGPT:
    def __init__(self, config):
        self.config = config
        self.messages = []
        self.engineName = None
        self.setAPICredentials()

    def setAPICredentials(self):
        # Setting the API key to use the OpenAI API
        openai.api_key = self.config.openAIParams['KEY']
        openai.api_base = self.config.openAIParams['BASE']
        openai.api_version = self.config.openAIParams['VERSION']
        openai.api_type = self.config.openAIParams['TYPE']
        self.engineName = self.config.openAIParams['DEPLOYMENT_NAME']

    def get_completion(self, prompt, callMaxLimit=5):
        messages = [{"role": "user", "content": prompt}]

        callComplete = False
        callAttemptCount = 0
        while not callComplete and callAttemptCount<callMaxLimit:
            try:
                response = openai.ChatCompletion.create(
                    engine=self.engineName,
                    messages=messages,
                    temperature=0
                )
                time.sleep(1)
                callComplete = True

            except openai.error.AuthenticationError as e:
                logging.error(f'Error Message: {e}')
                logging.error('Failed to send message. Trying again.')
                callComplete = False
            except openai.error.RateLimitError as e:
                logging.error(f'Error Message: {e}')
                logging.error('Rate limit hit. Pausing for a minute.')
                time.sleep(60)
                callComplete = False
            except openai.error.Timeout as e:
                logging.error(f'Error Message: {e}')
                logging.error('Timed out. Pausing for a minute.')
                time.sleep(60)
                callComplete = False
            except openai.error.InvalidRequestError as e:
                logging.error(f'Error Message: {e}')
                return e, False
            except Exception as e:
                logging.error(f'Error Message: {e}')
                logging.error('Failed to send message. Trying again.')
                callComplete = False
            callAttemptCount+=1 

        if callAttemptCount>=callMaxLimit:
            logging.error(f'Failed to send message at max limit of {callMaxLimit} times.')
            sys.exit('Exiting due to too many failed attempts.')

        elif callComplete:
            responseDict = {'Text':response.choices[0].message["content"], 
                            'Tokens':response.usage['total_tokens']}
            return responseDict, True

        