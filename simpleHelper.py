import os
import pypandoc
from pdf2docx  import parse
import pandas as pd
import numpy as np
import json
import openai
#from openai import AzureOpenAI
import os
from dotenv import load_dotenv

import time
import sys
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams  
import scipy.stats
import math
import textwrap
import re

#-------------------------------------------------------------------------------------------
# Code for Part 1


def makeBlankCriterionFile(pathToAssgnCSV, pathToGradingsCSV, pathToRubricsCSV, pathToCriterionCSV):
    gradeRubricAssignmentDF = getGRAData(pathToAssgnCSV, pathToGradingsCSV, pathToRubricsCSV)

    rubricData = gradeRubricAssignmentDF[['assignment_id', 'rubric_id', 'assignment_title', 'data_rubric']]\
                    .drop_duplicates(subset=['assignment_id', 'rubric_id']).sort_values('assignment_id').reset_index(drop=True)

    criterionList = []
    for index, row in rubricData.iterrows():
        rubricDict = {param: row[param] for param in ['assignment_id', 'rubric_id', 'assignment_title']}
        for criteria in row['data_rubric']:
            criteriaDict = {param: row[param] for param in rubricDict}
            for param in ['id', 'points', 'ratings', 'description', 'long_description']:
                criteriaDict[param] = criteria[param]
            criterionList.append(criteriaDict)

    customRubricTemplateDF = pd.DataFrame(criterionList)
    customRubricTemplateDF['custom_description'] = None

    if os.path.exists(pathToCriterionCSV):
        print('File already exists. Not overwriting. Change file path to save elsewhere.')
    else:
        customRubricTemplateDF.to_csv(pathToCriterionCSV, index=False)
        print(f'Saving criterion CSV to {pathToCriterionCSV}')


def convertSubmissions(originalSubmissionFolder, assignmentID):
    textSubmissionsFolder = os.path.join('data', f'Converted submissions_{assignmentID}')
    tempFolder = 'temp'
    if not os.path.exists(tempFolder):
        os.mkdir(tempFolder)
    if not os.path.exists(textSubmissionsFolder):
        os.mkdir(textSubmissionsFolder)

    for submissionFile in os.listdir(originalSubmissionFolder):
        submissionFilePath = os.path.join(originalSubmissionFolder, submissionFile)
        fileFormat = submissionFile.split('.')[-1].lower()

        if 'LATE' in submissionFile:
            userID = submissionFile.split('_')[2]
        else:
            userID = submissionFile.split('_')[1]

        savedFileName = userID+'.txt'
        savedFilePath = os.path.join(textSubmissionsFolder, savedFileName)
        if os.path.exists(savedFilePath):
            continue
        print(submissionFilePath)
        
        try:
            if fileFormat=='docx':
                output = pypandoc.convert_file(submissionFilePath, 'plain')
            elif fileFormat=='pdf':
                tempFilePath = os.path.join(tempFolder, 'tempFile.docx')
                parse(submissionFilePath, tempFilePath)
                output = pypandoc.convert_file(tempFilePath, 'plain')
        
            if len(output.split('\n')) < 16:
                print('File seems to have no text content in it. Skipping file.')
            else:
                with open(savedFilePath, 'w') as textFile:
                    textFile.write(output)
        except Exception as e:
            print(f'Error in conversion: {e}')
            print('Skipping file.')
            continue
    return True


#-------------------------------------------------------------------------------------------
# Code for Part 2


def getGRAData(pathToAssgnCSV, pathToGradingsCSV, pathToRubricsCSV):
    dataDFs = {}

    dataDFs['assignments'] = pd.read_csv(pathToAssgnCSV).drop_duplicates()
    dataDFs['gradings'] = pd.read_csv(pathToGradingsCSV).drop_duplicates()
    dataDFs['rubrics'] = pd.read_csv(pathToRubricsCSV).drop_duplicates()

    dataDFs['gradings'] = dataDFs['gradings'][dataDFs['gradings']['assessment_type']=='grading']
    dataDFs['gradings']['data'] = dataDFs['gradings']['data'].apply(lambda dataJSON: json.loads(dataJSON))
    dataDFs['rubrics']['data'] = dataDFs['rubrics']['data'].apply(lambda dataJSON: json.loads(dataJSON))

    if 'cleaned_description' not in dataDFs['assignments']:
        print('Column "cleaned_description" not found in assignments csv file. No assignment descriptions will be used.')
        dataDFs['assignments']['cleaned_description'] = ''

    if 'file_submission_source' in dataDFs['assignments']:
        print('Found custom assignment submission pointers.')
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
    
    return gradeRubricAssignmentDF


def checkIfSaved(assignmentID, userID, saveFolder, errorFolder):
    saveName = f"{assignmentID}-{userID}.p"
    for folder in [saveFolder, errorFolder]:
        filePath = os.path.join(folder, saveName)
        if os.path.exists(filePath):
            return True
    return False

def getRowCriterionDF(row, customDescMode, critDescDF):
    gradeDataDF = pd.DataFrame(row['data_grade'])
    rubricDataDF = pd.DataFrame(row['data_rubric'])

    fullCriterionDF = gradeDataDF.merge(rubricDataDF, left_on='criterion_id', 
                                        right_on='id', suffixes=('_grade', '_rubric'))

    if customDescMode:
        descDataDF = critDescDF[critDescDF['assignment_id']==row['assignment_id']]
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

def buildCritPrompt(criterionDF, useCustomDesc=True):
    fullCritText = ''
    for index, cRow in criterionDF.iterrows():
        if useCustomDesc:
            criteriaText = f"{index+1}. Criterion Title: '{cRow['description_rubric']}', \
CriterionID: '{cRow['criterion_id'] }', \
Max Points: '{cRow['points_rubric'] }', \
\nCriterion Guidelines: {cRow['custom_description']}\n"
            
        else:
            ratingsTextList = []
            for rating in cRow['ratings']:
                if rating['long_description']:
                    ratingsTextList += [f'\t{rating["long_description"]} : {rating["points"]} points\n']
                else:
                    ratingsTextList += [f'\t{rating["description"]} : {rating["points"]} points\n']

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

def getSubmissionText(assignmentID, userID, fileSource=1):
    if fileSource != -1:
        submissionFilePath = os.path.join('data', f'Converted submissions_{assignmentID}', f'{userID}.txt')
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
    

def promptBuilder(promptVariablesDict=None, saveTemplate=False, useCustomDesc=True):
    if saveTemplate:
        promptVariableNames = [
                    'Course Name',
                    'Assignment Name',
                    'Assignment Description',
                    'Student Submission',
                    'Criterion Description and Rubric',
                    'Maximum Points',
                    ]
        promptVariablesDict = {promptVariable:f'<<ENTER {promptVariable.upper()} HERE>>' for promptVariable in promptVariableNames}

    starterText = f'''You are a grader for the course "{promptVariablesDict['Course Name']}". 
    Your task is to grade a student's submission for the assignment "{promptVariablesDict['Assignment Name']}" using the provided criteria in the context of this course. 
    You will follow these specific rubric criteria to assign points related to different aspects of the assignment. '''

    if promptVariablesDict['Assignment Description']:
        assgnSummaryText = f"The assignment's summary is \"{promptVariablesDict['Assignment Description']}\". "
    else:
        assgnSummaryText = ''
    
    if useCustomDesc:
        guideText = '''Each criterion has guidelines used to grade that will inform you on how to make penalties and leave feedback. 
        Use the guidelines per criteria to assign a criteria score and feedback. '''
    else:
        guideText = '''Each criterion has a description of the criteria used to grade, and a ratings guide of points for reference which uses the format of <rating description> : <points>. 
        Use the ratings guide to assign points between 0 and the max points listed for each criteria. '''

    criterionStartText = f'''The points assigned must lie between 0 and the max points as listed for each criterion.
    The student's submission is delimited by triple backticks.
    The criteria are:
    '''


    
    endText = f'''The student submission is:
    ```{promptVariablesDict['Student Submission']}```
    For each criterion listed, return the assigned score, and feedback comment of under 100 words based on the criterion guidelines and errors made,
    Output a JSON object with one element ```peerGPT_grading``` to contain a array of JSON objects, with three fields: 
    peerGPT_criterion_id, peerGPT_criterion_score, peerGPT_reason
    '''
    fullText = starterText + assgnSummaryText + guideText + criterionStartText + promptVariablesDict['Criterion Description and Rubric'] + endText
    return fullText


# Processes the response generated by the GPT-3 model and extracts relevant information from it.
def processResponse(responseText):
    critScores = []
    '''responseLines = re.split('\n+', responseText)
    for line in responseLines:
        print(line)
        if ':' in line:
            if len(line.split(':'))==3:
                id, score, reason = line.split(':')
                score = float(score.strip().split()[0].strip())
                critScores.append({'peerGPT_criterion_id':id.strip(), 'peerGPT_criterion_score':score, 'peerGPT_reason':reason.strip()})
        else:
            critScores[-1]['peerGPT_reason'] += line
            '''
    response_json = json.loads(responseText)
    if "peerGPT_grading" in response_json:
        critScores = response_json["peerGPT_grading"]
    return pd.DataFrame(critScores)


def saveOutputasPickle(dataDict, runSuccess, saveFolder, errorFolder):
    saveName = f"{dataDict['assignment_id']}-{dataDict['submitter_id']}.p"
    pickleFolder = saveFolder if runSuccess else errorFolder
    
    pickleSavePath = os.path.join(pickleFolder, saveName)

    with open(pickleSavePath, 'wb') as handle:
        pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True


def processGRARow(row, courseName, customDescMode, critDescDF):        
    fullCriterionDF = getRowCriterionDF(row, customDescMode, critDescDF)
    fullCritText = buildCritPrompt(fullCriterionDF)
    studentSubmissionStatus, studentSubmission = getSubmissionText(row['assignment_id'], row['submitter_id'])
    
    if studentSubmission:
        promptVariableDict = {
                            'Course Name': courseName,
                            'Assignment Name': row['assignment_title'],
                            'Assignment Description': row['cleaned_description'] \
                                                        if row['cleaned_description'] else None,
                            'Student Submission': studentSubmission,
                            'Criterion Description and Rubric': fullCritText,
                            'Maximum Points': row['points_possible'],
                            }
        fullPrompt = promptBuilder(promptVariableDict, useCustomDesc=customDescMode)

        peerBot = peerGPT()
        response, responseSucess = peerBot.get_completion(fullPrompt)
        del peerBot
        
        if responseSucess:
            print(response)
            scoreBreakdownDF = processResponse(response['Text'])
            print(scoreBreakdownDF)

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
    


class peerGPT:
    def __init__(self):
        self.messages = []
        self.engineName = None

    def get_completion(self, prompt, callMaxLimit=5):
        #Sets the current working directory to be the same as the file.
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        #Load environment file for secrets.
        try:
            if load_dotenv('.env') is False:
                raise TypeError
        except TypeError:
            print('Unable to load .env file.')
            quit()
        #Create Azure client
        client = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],  
            api_version=os.environ['OPENAI_API_VERSION'],
            azure_endpoint = os.environ['OPENAI_API_BASE'],
            organization = os.environ['ORGANIZAION']
        )

        messages = [{"role": "user", "content": prompt}]

        callComplete = False
        callAttemptCount = 0
        while not callComplete and callAttemptCount<callMaxLimit:
            try:
                response = client.chat.completions.create(
                    model=os.environ['OPENAI_MODEL'],
                    messages=messages,
                    temperature=0,
                    # timeout=60,
                )
                time.sleep(1)
                callComplete = True

            except OpenAI.error.AuthenticationError as e:
                print(f'Error Message: {e}')
                print('Failed to send message. Trying again.')
                callComplete = False
            except openai.error.RateLimitError as e:
                print(f'Error Message: {e}')
                print('Rate limit hit. Pausing for a minute.')
                time.sleep(60)
                callComplete = False
            except openai.error.Timeout as e:
                print(f'Error Message: {e}')
                print('Timed out. Pausing for a minute.')
                time.sleep(60)
                callComplete = False
            except openai.error.InvalidRequestError as e:
                print(f'Error Message: {e}')
                return e, False
            except Exception as e:
                print(f'Error Message: {e}')
                print('Failed to send message. Trying again.')
                callComplete = False
            callAttemptCount+=1 

        if callAttemptCount>=callMaxLimit:
            print(f'Failed to send message at max limit of {callMaxLimit} times.')
            sys.exit('Exiting due to too many failed attempts.')

        elif callComplete:
            responseDict = {'Text':response.choices[0].message.content, 
                            'Tokens':response.usage.total_tokens}
            return responseDict, True
        

#-------------------------------------------------------------------------------------------
# Code for Part 3


def convertPicklesToDF(pickleFolder):
    saveDataList = []
    for savedPickle in os.listdir(pickleFolder):
        if '.DS_Store' in savedPickle:
            # print('Skipping Mac DS_Store folder.')
            continue
        rowSaveData = pickle.load(open(os.path.join(pickleFolder, savedPickle), 'rb'))
        saveDataList.append(rowSaveData)
    resultsDF = pd.DataFrame(saveDataList)
    return resultsDF

def getCriterionDataDF(resultsDF, saveName, excelFolder):
    mergedCriterionData = pd.DataFrame()
    for index,row in resultsDF.iterrows():
        criterionData = row['data_peerGPT']
        for col in ['submitter_id', 'assignment_id']:
            criterionData[col] = row[col]
        mergedCriterionData = pd.concat([mergedCriterionData, criterionData])

    mergedCriterionData.to_excel(os.path.join(excelFolder, saveName+'-CriterionData.xlsx'))

    saveDF = resultsDF.copy()
    #del saveDF['data_peerGPT']
    saveDF.to_excel(os.path.join(excelFolder, saveName+' - ScoreData.xlsx'))
    
    return mergedCriterionData

def getScoreSpread(resultsDF, saveName, chartFolder):
    sns.set_theme(style="whitegrid", palette="deep")

    filterDF = resultsDF
    maxScore = resultsDF['points_possible'].max()
    assgnCount = len(resultsDF['assignment_id'].unique())

    asgmtList = resultsDF['assignment_id'].unique().tolist()
    asgmtNames = {asgmt: f'Asgn. {asgmtList.index(asgmt)+1}' for asgmt in asgmtList}
    filterDF['Assignment'] = filterDF['assignment_id'].apply(lambda asgmt: asgmtNames[asgmt])

    sns.jointplot(data=filterDF, x='score', y='peerGPT_score', hue='Assignment', height=5, marker=".", s=50, palette=sns.color_palette()[:assgnCount])
    plt.plot([0,maxScore],[0,maxScore], lw=1, color='#313232', linestyle='dashed')
    # plt.plot([1,46],[0,40], lw=1, color='#aaaaaa', linestyle='dashed')
    # plt.plot([0,40],[1,46], lw=1, color='#aaaaaa', linestyle='dashed')g.set_xlabel('Grader Score',fontsize=8)
    plt.xlabel('Grader Score', fontsize=12)
    plt.ylabel('peerGPT Score', fontsize=12, rotation=90)
    plt.legend(title='Assignment', fontsize=8)
    # plt.show()
    plt.savefig(os.path.join(chartFolder, f'{saveName} - JointPlot.png'), dpi=300, bbox_inches='tight')
    # plt.close()
    return True

def saveGraderPeerGPTMeanScoreDiff(resultsDF, saveName, excelFolder):
    excludeDF = resultsDF.copy()
    excludeDF['Score Difference'] = excludeDF['peerGPT_score']-excludeDF['score']
    excludeDF['Score Diff. %'] = 100*(excludeDF['peerGPT_score']-excludeDF['score'])/excludeDF['points_possible']

    meanDiffDict = {}
    meanDiffPercentDict = {}
    for group in excludeDF.groupby(['grader_id','assignment_id']):
        if group[0][0] not in meanDiffDict:
            meanDiffDict[group[0][0]] = {}
            meanDiffPercentDict[group[0][0]] = {}
        meanDiffDict[group[0][0]][group[0][1]] = group[1]["Score Difference"].mean()
        meanDiffPercentDict[group[0][0]][group[0][1]] = np.round(group[1]["Score Diff. %"].mean() , 2)
    meanDiffDF = pd.DataFrame(meanDiffDict)
    meanDiffPercentDF = pd.DataFrame(meanDiffPercentDict)

    print(f"Saving file at: {os.path.join(excelFolder, saveName+' - Grader - peerGPT Score Difference.xlsx')}")
    meanDiffDF.to_excel(os.path.join(excelFolder, saveName+' - Grader - peerGPT Score Difference.xlsx'))

    print(f"Saving file at: {os.path.join(excelFolder, saveName+' - Grader - peerGPT Score Diff. %.xlsx')}")
    meanDiffPercentDF.to_excel(os.path.join(excelFolder, saveName+' - Grader - peerGPT Score Diff. %.xlsx'))
    return meanDiffDF, meanDiffPercentDF


def buildFullInfoDF(GRADataDF, resultsDF, saveName, excelFolder):
    print("sss")
    critDataDF = pd.DataFrame()
    for index,row in resultsDF.iterrows():
        criterionData = row['data_peerGPT']
        for col in ['submitter_id', 'assignment_id', 'grader_id']:
            criterionData[col] = row[col]
        critDataDF = pd.concat([critDataDF, criterionData])
    allCritDF = critDataDF.drop(['mastery_points','ignore_for_scoring','title','peerGPT_criterion_id','description_grade'],
                            axis=1, errors='ignore')
    
    print(allCritDF)

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

    assignmentDF = GRADataDF[['assignment_id', 'assignment_title']].drop_duplicates()
    baseInfoDF = allCritDF[['assignment_id', 'criterion_id', 'description_rubric', 'points_rubric']].drop_duplicates()
    baseInfoDF = baseInfoDF.merge(assignmentDF, on='assignment_id')

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

    fullInfoDF['Grader Mean Diff. %'] = 100*(fullInfoDF['All Graders Mean'] - fullInfoDF['Grader Mean']).div(fullInfoDF['points_rubric'])

    print(f"sSaving file at: {os.path.join(excelFolder, saveName+' - Grader Difference Table.xlsx')}")
    fullInfoDF.to_excel(os.path.join(excelFolder, saveName+' - Grader Difference Table.xlsx'))

    rubricInfo = GRADataDF[['assignment_id', 'data_rubric']].drop_duplicates('assignment_id').reset_index(drop=True)
    rubricOrderDict = {}
    for index, row in rubricInfo.iterrows():
        rubricOrderDict[row['assignment_id']] = pd.DataFrame(row['data_rubric'])['description'].tolist()

    return fullInfoDF, rubricOrderDict


def confindenceInterval(data, confidence=0.9):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


# Generates Z-Score and Confidence Interval info for all graders against themselves and GraderGPT.
def getZScoreAndCI(fullInfoDF, saveName, excelFolder, confidence=0.93):
    zScoreGraderList, zScoreGPTList = [], []
    CIGraderList, CIGPTList = [], []

    for AID in fullInfoDF['assignment_id'].unique():
        for CID in fullInfoDF[(fullInfoDF['assignment_id']==AID)]['criterion_id'].unique(): 
            subsetDF =  fullInfoDF[(fullInfoDF['assignment_id']==AID) & (fullInfoDF['criterion_id']==CID)]
            zScoresGrader = list(scipy.stats.zscore(subsetDF['Grader Mean Diff. %']))
            zScoresGPT = list(scipy.stats.zscore(subsetDF['Mean Difference %']))
            zScoreGraderList += zScoresGrader
            zScoreGPTList += zScoresGPT

            meanGraderDiffDict = dict(zip(subsetDF['grader_id'].tolist(),subsetDF['Grader Mean Diff. %'].tolist()))
            meanGPTDiffDict = dict(zip(subsetDF['grader_id'].tolist(),subsetDF['Mean Difference %'].tolist()))

            lowerGrader,upperGrader = confindenceInterval(list(meanGraderDiffDict.values()), confidence)
            lowerGPT,upperGPT = confindenceInterval(list(meanGPTDiffDict.values()), confidence)
            for grader in meanGraderDiffDict:
                if meanGraderDiffDict[grader]<lowerGrader or meanGraderDiffDict[grader]>upperGrader:
                    CIGraderList.append('Out of CI')
                else:
                    CIGraderList.append('Within CI')
                
                if meanGPTDiffDict[grader]<lowerGPT or meanGPTDiffDict[grader]>upperGPT:
                    CIGPTList.append('Out of CI')
                else:
                    CIGPTList.append('Within CI')

    ZScoreInfoDF = fullInfoDF.copy()
    ZScoreInfoDF['Z-Score against GraderGPT'] = zScoreGPTList
    ZScoreInfoDF['Z-Score b/w Graders'] = zScoreGraderList
    
    ZScoreInfoDF[f'CI using GraderGPT with Confidence={confidence}'] = CIGPTList
    ZScoreInfoDF[f'CI using Graders with Confidence={confidence}'] = CIGraderList
    
    ZScoreInfoDF = ZScoreInfoDF.drop(['Mean Difference', 'Grader Std. Dev.', 'peerGPT Std. Dev.', 'All Graders Std. Dev.', 'Global peerGPT Std. Dev.'], axis=1)
    ZScoreInfoDF = ZScoreInfoDF.rename(columns={'Mean Difference %':'Mean Diff. % against GraderGPT', \
                                            'Grader Mean Diff. %':'Mean Diff. % b/w Graders', \
                                            'peerGPT Mean':'GraderGPT Mean', \
                                            'assignment_title':'Title', \
                                            'description_rubric':'Rubric', \
                                            'points_rubric':'Max Score'}).reset_index(drop=True)
    ZScoreInfoDF = ZScoreInfoDF.set_index(['assignment_id', 'Title', 'criterion_id', 'Rubric', 
                                        'All Graders Mean', 'Global peerGPT Mean', 'Max Score',
                                        'grader_id'])
    
    print(f"Saving file at: {os.path.join(excelFolder, saveName+f' - Z-Score & CI@{confidence} Details.xlsx')}")
    ZScoreInfoDF.to_excel(os.path.join(excelFolder, saveName+f' - Z-Score & CI@{confidence} Details.xlsx'))
    
    return ZScoreInfoDF


# Generates strip plots to visualize the percentage mean difference between grader scores and peerGPT scores for each criterion. 
# The plots are saved in a folder named 'Mean Diff %' within the specified chartFolder.
def getMeanDiffPercentCharts(fullInfoDF, rubricOrderDict, saveName, chartFolder):
    sns.set_theme(style="darkgrid") #, palette="dark")
    saveMeanFolder = os.path.join(chartFolder, 'Mean Diff %')
    if not os.path.exists(saveMeanFolder):
        os.mkdir(saveMeanFolder)

    for AID in fullInfoDF['assignment_id'].unique():
        subsetDF =  fullInfoDF[fullInfoDF['assignment_id']==AID]
        plt.clf() 

        graderCount = len(subsetDF['grader_id'].unique())

        upperY = math.ceil(max(subsetDF['Mean Difference %']))
        lowerY = math.floor(min(subsetDF['Mean Difference %']))
        if upperY<0:
            upperY = 0
        if lowerY>0:
            lowerY = 0
        if upperY-lowerY>160:
            tickStep = 10
        else:
            tickStep = 5
        if lowerY<0 and upperY>0:
            tickSpace = np.concatenate((np.arange(lowerY-lowerY%tickStep,0, tickStep),np.arange(0,upperY+upperY%tickStep+5, tickStep)))
        else:
            tickSpace = np.arange(lowerY,upperY+10, tickStep)

        sns.set(rc={'figure.figsize':((3/2)*len(rubricOrderDict[AID]),3)})

        g = sns.stripplot(data=subsetDF, x='description_rubric', y='Mean Difference %', \
                            order = rubricOrderDict[AID], \
                            hue='grader_id', dodge=False, jitter=True, \
                            palette=sns.color_palette(n_colors=graderCount)[:graderCount])
        
        plt.axhline(y=0, color='#313232', linestyle='--')

        g.set_ylim(lowerY-5,upperY+5)
        g.set_yticks(tickSpace)
        g.set_xlabel('Criteria', fontsize=12)
        g.set_ylabel('Mean Difference %', fontsize=12, rotation=90)

        g.set_xticks(g.get_xticks())

        wrapSize = 14 if len(rubricOrderDict[AID]) < 6 else 12 
        g.set_xticklabels([textwrap.fill(t.get_text(), wrapSize, break_long_words=False) \
                           for t in g.get_xticklabels()], size=9)

        g.set_title(textwrap.fill(subsetDF['assignment_title'].iloc[0], 50))

        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, title='Grader ID')
        # plt.show()
        g.get_figure().savefig(os.path.join(saveMeanFolder, f'{saveName} - {AID} - MeanDiffSpread.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Charts saved at: {saveMeanFolder}")
    return True