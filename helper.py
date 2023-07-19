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
    dataDFs = {}
    for dataFileName in ['assignments', 'gradings', 'rubrics']:
        fullFileName = f'{config.courseName}{dataFileName}.csv'
        filePath = os.path.join(config.baseDataFolder, config.dataFolders['CSV_DATA'], fullFileName)
        dataDF = pd.read_csv(filePath).drop_duplicates()
        dataDFs[dataFileName] = dataDF

    dataDFs['gradings'] = dataDFs['gradings'][dataDFs['gradings']['assessment_type']=='grading']
    dataDFs['gradings']['data'] = dataDFs['gradings']['data'].apply(lambda dataJSON: json.loads(dataJSON))
    dataDFs['rubrics']['data'] = dataDFs['rubrics']['data'].apply(lambda dataJSON: json.loads(dataJSON))

    if 'cleaned_description' not in dataDFs['assignments']:
        logging.warn('Column "cleaned_description" not found in assignments csv file. No assignment descriptions will be used.')
        dataDFs['assignments']['cleaned_description'] = ''

    if 'file_submission_source' in dataDFs['assignments']:
        logging.info('Found custom assignment submission pointers.')
        dataDFs['assignments']['file_submission_source'] = dataDFs['assignments']['file_submission_source'].astype('Int64').fillna(-1)
    else:
        dataDFs['assignments']['file_submission_source'] = dataDFs['assignments']['assignment_id'].astype('Int64').fillna(-1)
    
    gradeRubricDF = dataDFs['gradings'].merge(dataDFs['rubrics'], on='rubric_id', \
                                              suffixes=('_grade', '_rubric'))
    gradeRubricAssignmentDF = gradeRubricDF.merge(dataDFs['assignments'], on='assignment_id', \
                                                  suffixes=('', '_assignment'))

    gradeRubricAssignmentDF = gradeRubricAssignmentDF[['submitter_id', 'grader_id', 'score', 'rubric_id', 
                                                       'assignment_id', 'assignment_title', 'data_grade', 
                                                       'data_rubric', 'points_possible', 'assignment_description', 
                                                       'cleaned_description', 'file_submission_source']]
    if mpMode:
        rowDataList = []
        for index, row in gradeRubricAssignmentDF.iterrows():
            rowDataList.append({'row':row, 'config':config})
        return rowDataList
    else:    
        return gradeRubricAssignmentDF


def promptBuilder(promptVariablesDict=None, saveTemplate=False, config=None, useCustomDesc=True):
    if saveTemplate:
        promptVariablesDict = {promptVariable:f'<<ENTER {promptVariable.upper()} HERE>>' for promptVariable in config.promptVariableNames}

    starterText = f'''You are a grader for the course "{promptVariablesDict['Course Name']}". 
Your task is to grade a student's submission for the assignment "{promptVariablesDict['Assignment Name']}" using the provided criteria in the context of this course. 
You will follow these specific rubric criteria to assign points related to different aspects of the assignment.'''

    if promptVariablesDict['Assignment Description']:
        assgnSummaryText = f"The assignment's summary is \"{promptVariablesDict['Assignment Description']}\"."
    else:
        assgnSummaryText = ''
    
    if useCustomDesc:
        guideText = '''Each criteria has guidelines used to grade that will inform you on how to make penalties and leave feedback. 
Use the guidelines per criteria to assign a criteria score and feedback.'''
    else:
        guideText = '''Each criterion has a description of the criteria used to grade, and a ratings guide of points for reference which uses the format of <rating description> : <points>. 
Use the ratings guide to assign points between 0 and the max points as listed for each criteria.'''

    criterionStartText = f'''The points assigned must lie between 0 and the max points as listed for each criteria.
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
    fullText = starterText + assgnSummaryText + guideText + criterionStartText + promptVariablesDict['Criterion Description and Rubric'] + endText
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
                criteriaText = f"{index+1}. Criterion Title: '{cRow['description_rubric']}', \
CriterionID: '{cRow['criterion_id'] }', \
Max Points: '{cRow['points_rubric'] }', \
\nCriterion Description: '{cRow['long_description']}', \
\nRatings Guide:\n"+''.join(ratingsTextList)
                
            else:
                criteriaText = f"{index+1}. Criterion Title: '{cRow['description_rubric']}', \
CriterionID: '{cRow['criterion_id'] }', \
Max Points: '{cRow['points_rubric'] }', \
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

def getSubmissionText(assignmentID, userID, fileSource, config):
    if fileSource != -1:
        submissionFilePath = os.path.join(config.baseDataFolder, \
                                        config.dataFolders['TEXT_SUBMISSIONS'], \
                                            f'{fileSource}', f'{userID}.txt')
        if os.path.exists(submissionFilePath):
            try:
                with open(submissionFilePath, errors="ignore") as textFile:
                    submissionLines = textFile.readlines()
                studentSubmission = ''.join([line.strip() if line!='\n' else '\n' for line in submissionLines])
                studentSubmission = studentSubmission.replace('\n\n', '\n')
                return True, studentSubmission
            except Exception as e:
                return 'Could not open the file', False
        else:
            return 'Submission file missing.', False
    else:
        return 'Not a gradable assignment', False

def checkIfSaved(assignmentID, userID, config):
    saveName = f"{assignmentID}-{userID}.p"
    for pickleFolder in ['saves', 'errors']:
        filePath = os.path.join(config.baseOutputFolder, \
                                config.outputPickleLookup[pickleFolder], \
                                config.fullName, \
                                saveName)
        if os.path.exists(filePath):
            return True
    return False

def getRowCriterionDF(row, config):
    gradeDataDF = pd.DataFrame(row['data_grade'])
    rubricDataDF = pd.DataFrame(row['data_rubric'])

    fullCriterionDF = gradeDataDF.merge(rubricDataDF, left_on='criterion_id', 
                                        right_on='id', suffixes=('_grade', '_rubric'))

    if config.customDescMode:
        descDataDF = config.critDescDF[config.critDescDF['assignment_id']==row['assignment_id']]
        descDataDF = descDataDF[['custom_description', 'id']]
        fullCriterionDF = fullCriterionDF.merge(descDataDF, left_on='criterion_id', right_on='id')

    if 'points' not in gradeDataDF.columns:
        fullCriterionDF['points_rubric'] = fullCriterionDF['points']
        fullCriterionDF['points_gradec'] = fullCriterionDF['points']

    fullCriterionDF = fullCriterionDF.drop(['id_grade', 'id_rubric', 'learning_outcome_id', 'id', 'points'
                                            'comments_enabled', 'comments_html', 'criterion_use_range'], 
                                            axis=1, errors='ignore') 
    # fullCriterionDF = fullCriterionDF.rename(columns={'points':})
    return fullCriterionDF

def processTokenCount(row, config):        
    fullCriterionDF = getRowCriterionDF(row, config)
    fullCritText = buildCritPrompt(fullCriterionDF, useCustomDesc=config.customDescMode)
    studentSubmissionStatus, studentSubmission = getSubmissionText(row['assignment_id'], row['submitter_id'], row['file_submission_source'], config)

    if studentSubmission:
        promptVariableDict = {
                            'Course Name': config.simpleCourseName,
                            'Assignment Name': row['assignment_title'],
                            'Assignment Description': row['cleaned_description'] \
                                                        if row['cleaned_description'] else None,
                            'Student Submission': studentSubmission,
                            'Criterion Description and Rubric': fullCritText,
                            'Maximum Points': row['points_possible'],
                            }
        fullPrompt = promptBuilder(promptVariableDict, useCustomDesc=config.customDescMode)
        # print(fullPrompt)
        return fullPrompt
    else:
        return False

def processGRARow(row, config):        
    fullCriterionDF = getRowCriterionDF(row, config)
    fullCritText = buildCritPrompt(fullCriterionDF, useCustomDesc=config.customDescMode)
    studentSubmissionStatus, studentSubmission = getSubmissionText(row['assignment_id'], row['submitter_id'], row['file_submission_source'], config)
    
    if studentSubmission:
        promptVariableDict = {
                            'Course Name': config.simpleCourseName,
                            'Assignment Name': row['assignment_title'],
                            'Assignment Description': row['cleaned_description'] \
                                                        if row['cleaned_description'] else None,
                            'Student Submission': studentSubmission,
                            'Criterion Description and Rubric': fullCritText,
                            'Maximum Points': row['points_possible'],
                            }
        fullPrompt = promptBuilder(promptVariableDict, useCustomDesc=config.customDescMode)
       
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
                'Error':studentSubmissionStatus}, False
    
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
        dataDict, runSuccess = processGRARow(rowData['row'], rowData['config'])
        saveOutputasPickle(dataDict, runSuccess, rowData['config'])
        return True

def saveOutputasPickle(dataDict, runSuccess, config):
    saveName = f"{dataDict['assignment_id']}-{dataDict['submitter_id']}.p"

    pickleFolder = 'saves' if runSuccess else 'errors'
    pickleSavePath = os.path.join(config.baseOutputFolder, \
                                    config.outputPickleLookup[pickleFolder], \
                                    config.fullName, \
                                    saveName)

    with open(pickleSavePath, 'wb') as handle:
        pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def convertPicklesToDF(folderName, config):
    saveDataList = []
    pickleFolder = os.path.join(config.baseOutputFolder, \
                                config.outputPickleLookup[folderName], \
                                config.fullName)
    for savedPickle in os.listdir(pickleFolder):
        rowSaveData = pickle.load(open(os.path.join(pickleFolder, savedPickle), 'rb'))
        # if folderName=='saves':
        #     if 'peerGPT_criterion_score' in rowSaveData['data_peerGPT']:
        #         rowSaveData['peerGPT_score_real'] = rowSaveData['data_peerGPT']['peerGPT_criterion_score'].sum()
        #     else:
        #         rowSaveData['peerGPT_score_real'] = rowSaveData['data_peerGPT']['peerGPT__criterion_score'].sum()
        saveDataList.append(rowSaveData)
    resultsDF = pd.DataFrame(saveDataList)
    return resultsDF

def checkFolderData(folderPath):
    folderExists, dataExists = True, True
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
        folderExists = False
        dataExists = False
    elif not len(os.listdir(folderPath)):
        dataExists = False

    return folderExists, dataExists


class Config:
    def __init__(self):
        self.logLevel = logging.INFO

        self.openAIParams: dict = {'KEY': None,
                                   'BASE':None,
                                   'VERSION':None,
                                   'TYPE':None,
                                   'DEPLOYMENT_NAME':None}
        
        self.baseDataFolder: str = 'data'
        self.baseOutputFolder: str = 'output'
        self.tempFolder: str = 'temp'
        self.promptFolder: str = 'prompts'

        self.dataFolders: dict = {
                                'SUBMISSIONS':'Submissions',
                                'CSV_DATA':'CSV Data',
                                'TEXT_SUBMISSIONS':'Converted Text Submissions',
                                'SUPPLEMENTAL_FILES':'Supplemental Documentation'
                                }
        
        self.outputFolders: dict = {
                                'EXCEL_OUTPUT':'Excel File Outputs',
                                'CHART_OUTPUT':'Chart File Outputs',
                                'PICKLE_SAVES':'Pickled Save Outputs',
                                'PICKLE_ERRORS':'Pickled Error Outputs',
                                'PROMPT_FILES':'Saved Prompt Files',
                                }
        
        self.outputPickleLookup: dict = {
                                'saves':'Pickled Save Outputs',
                                'errors':'Pickled Error Outputs',
                                }
        
        self.CSVDataFiles = ['criterion', 'assignments', 'gradings', 'rubrics']

        self.courseNameLookup: dict = {
                                'MOVESCI': 'MOVESCI_110_WN_2023_584988_MWrite_', \
                                'ECON': 'ECON_101_300_WN_2023_566636_', \
                                }
        
        self.courseName: str = None
        self.simpleCourseName: str = None
        
        self.promptVariableNames = [
                    'Course Name',
                    'Assignment Name',
                    'Assignment Description',
                    'Student Submission',
                    'Criterion Description and Rubric',
                    'Maximum Points',
                    ]

        self.customDescMode = True
        self.overWriteSave: bool = False
        self.fullName: str = None
        self.critDescDF = None
        self.poolSize: int = 8

    def validateDataFolders(self):
        dataValidationTracker = True
        for dataFolder in self.dataFolders:
            dataFolderPath = os.path.join(self.baseDataFolder, self.dataFolders[dataFolder])
            folderExists, dataExists = checkFolderData(dataFolderPath)
            dataValidationSuccess = folderExists and dataExists
            if dataFolder not in ['SUPPLEMENTAL_FILES']:
                dataValidationTracker = False if not dataValidationSuccess or not dataValidationTracker else True
            if dataFolder in ['SUBMISSIONS', 'CSV_DATA']:
                if not folderExists:
                    errorMsg = f'No folder for "{dataFolderPath}". Please add data to folder.'
                    logging.error(errorMsg)
                if folderExists and not dataExists:
                    errorMsg = f'No data found in folder for "{dataFolderPath}". Please add data to folder.'
                    logging.error(errorMsg)
            elif dataFolder in ['TEXT_SUBMISSIONS']:
                if not folderExists or not dataExists:
                    errorMsg = f'No data found in "{dataFolderPath}". Run conversion script to generate text data first.'
                    logging.error(errorMsg)

        return dataValidationTracker

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

        self.courseName = self.configFetch(
            'FULL_COURSE_NAME', self.courseName, str)
        envImportSuccess = False if not self.courseName or not envImportSuccess else True

        self.simpleCourseName = self.configFetch(
            'SIMPLE_COURSE_NAME', self.simpleCourseName, str)
        envImportSuccess = False if not self.simpleCourseName or not envImportSuccess else True

        for folder in [self.baseDataFolder, self.baseOutputFolder, self.tempFolder, self.promptFolder]:
            checkFolderData(folder)

        envImportSuccess = self.validateDataFolders()

        for outputFolder in self.outputFolders:
            outputFolderPath = os.path.join(self.baseOutputFolder, self.outputFolders[outputFolder])
            checkFolderData(outputFolderPath)

        if not envImportSuccess:
            sys.exit('Exiting due to configuration parameter import problems.')
        else:
            logging.info('All configuration parameters set up successfully.')

    def setSaveDetails(self, fullName):
        self.fullName = fullName
        self.simpleCourseName = self.fullName.split('-')[0]
        self.courseName = self.courseNameLookup[self.simpleCourseName]
        for outputFolder in self.outputFolders:
            versionOutputFolderPath = os.path.join(self.baseOutputFolder, \
                                                   self.outputFolders[outputFolder], \
                                                   self.fullName)
            checkFolderData(versionOutputFolderPath)
        
        if self.customDescMode:
            filePath = os.path.join(self.baseDataFolder, \
                                    self.dataFolders['CSV_DATA'], \
                                    f'{self.courseName}criterion.csv')
            self.critDescDF = pd.read_csv(filePath)
        return True

    def saveTemplatePrompt(self):
        with open(os.path.join(self.baseOutputFolder, \
                               self.outputFolders['PROMPT_FILES'], \
                               self.fullName, \
                               f'{self.fullName}_templatePrompt.txt'), 'w') as textFile:
            textFile.write(promptBuilder(saveTemplate=True, config=self, useCustomDesc=self.customDescMode))
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
                    temperature=0,
                    # timeout=60,
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

        