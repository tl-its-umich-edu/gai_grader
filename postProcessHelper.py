import os
import textwrap
import math
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from helper import *

resultsDFCols = ['submitter_id', 'grader_id', 'rubric_id', 'assignment_id', 'score', 'points_possible']
criterionCommonCols = ['points_grade', 'criterion_id', 'description_grade', 'comments', \
                       'description_rubric', 'points_rubric', 'custom_description', \
                       'submitter_id', 'assignment_id', 'grader_id']

def getScoreSpread(resultsDF, chartFolder):
    sns.set_theme(style="whitegrid", palette="deep")

    filterDF = resultsDF#[resultsDF['assignment_id']!=1916709]
    maxScore = resultsDF['points_possible'].max()
    assgnCount = len(resultsDF['assignment_id'].unique())
    sns.jointplot(data=filterDF, x='score', y='peerGPT_score', hue='assignment_id', height=5, marker=".", s=50, palette=sns.color_palette()[:assgnCount])
    plt.plot([0,maxScore],[0,maxScore], lw=1, color='#313232', linestyle='dashed')
    # plt.plot([1,46],[0,40], lw=1, color='#aaaaaa', linestyle='dashed')
    # plt.plot([0,40],[1,46], lw=1, color='#aaaaaa', linestyle='dashed')g.set_xlabel('Grader Score',fontsize=8)
    plt.xlabel('Grader Score', fontsize=12)
    plt.ylabel('peerGPT Score', fontsize=12, rotation=90)
    plt.legend(title='Asgn. ID', fontsize=8)
    # plt.show()
    plt.savefig(os.path.join(chartFolder, 'JointPlot.png'), dpi=300, bbox_inches='tight')
    # plt.close()
    return True

def getCriterionDataDF(resultsDF, saveName, excelFolder):
    mergedCriterionData = pd.DataFrame()
    for index,row in resultsDF.iterrows():
        criterionData = row['data_peerGPT']
        for col in ['submitter_id', 'assignment_id']:
            criterionData[col] = row[col]
        mergedCriterionData = pd.concat([mergedCriterionData, criterionData])

    mergedCriterionData.to_excel(os.path.join(excelFolder, saveName+'-CriterionData.xlsx'))

    saveDF = resultsDF.copy()
    del saveDF['data_peerGPT']
    saveDF.to_excel(os.path.join(excelFolder, saveName+'-ScoreData.xlsx'))
    
    return mergedCriterionData

def getFullHistogramSpread(resultsDF, chartFolder):
    sns.set_theme(style="darkgrid", palette="dark")
    rcParams['figure.figsize'] = 5,3

    peerGPTGradesDF = resultsDF.copy().drop(['score', 'data_peerGPT'], axis=1, errors='ignore') 
    peerGPTGradesDF = peerGPTGradesDF.rename(columns={'peerGPT_score':'score'})
    peerGPTGradesDF['Grader Type'] = 'peerGPT'

    gradersDF = resultsDF.copy().drop(['peerGPT_score', 'data_peerGPT'], axis=1, errors='ignore') 
    gradersDF['Grader Type'] = gradersDF['grader_id'].apply(lambda id: f'Grader ID: {id}')

    allGradesDF = pd.concat([gradersDF, peerGPTGradesDF])

    fig, axes = plt.subplots(nrows=len(resultsDF['grader_id'].unique()), \
                             ncols=len(resultsDF['assignment_id'].unique()), \
                             figsize=(3*len(resultsDF['assignment_id'].unique()), \
                                      1.5*len(resultsDF['grader_id'].unique())), \
                             layout="constrained")

    for col, assignmentID in enumerate(resultsDF['assignment_id'].unique()):
        for row, graderID in enumerate(sorted(resultsDF['grader_id'].unique())):
            subsetDF = allGradesDF[(allGradesDF['assignment_id']==assignmentID) \
                                   & (allGradesDF['grader_id']==graderID)]
            # display(subsetDF)

            if not subsetDF.empty:
                upperX = int(subsetDF['points_possible'].iloc[0])
                minScore = min(resultsDF[(resultsDF['assignment_id']==assignmentID)]['score'])
                lowerX = int(minScore-minScore%2)
                xTickStep = 1 if upperX-lowerX < 10 else 2

                g = sns.histplot(ax=axes[row,col], data=subsetDF, x='score', \
                                hue='Grader Type', kde=True, multiple="dodge", \
                                palette=sns.color_palette()[:2])
                # g = sns.scatterplot(data=subsetDF, x='score', hue='Grader Type', palette=sns.color_palette()[:2])
                g.set_xlim(lowerX,upperX)
                g.set_xticks(range(lowerX,upperX+1, xTickStep))
                # g.set_title(f'Assignment ID: {assignmentID}')
                # g.set_xlabel('Points Awarded')
                g.set_xlabel('Score Distribution', fontsize=8)
                g.set_ylabel('')
                g.legend([],[], frameon=False)

    pad = 5
    for ax, col in zip(axes[0], resultsDF['assignment_id'].unique()):
        ax.set_title(f'Assignment ID: {col}')
    for ax, row in zip(axes[:,0], sorted(resultsDF['grader_id'].unique())):
            ax.annotate(f'Grader ID: {row}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)

    # plt.show()
    plt.savefig(os.path.join(chartFolder, 'HistogramPlotSpread.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return True

def getFullScatterplotSpread(resultsDF, chartFolder,outlierFactor = 0.15):
    sns.set_theme(style="darkgrid", palette="dark")
    rcParams['figure.figsize'] = 4,4

    gradesDF = resultsDF.copy()
    gradesDF['Score Difference'] = gradesDF['peerGPT_score']-gradesDF['score']
    gradesDF['Outlier'] = (gradesDF['Score Difference']/(gradesDF['points_possible']*outlierFactor)).apply(lambda scoreDiff: 'Outlier' if abs(scoreDiff)>1 else 'In range')

    fig, axes = plt.subplots(nrows=len(resultsDF['grader_id'].unique()), ncols=len(resultsDF['assignment_id'].unique()), \
                        figsize=(15,10), layout="constrained")

    for col, assignmentID in enumerate(resultsDF['assignment_id'].unique()):
        for row, graderID in enumerate(sorted(resultsDF['grader_id'].unique())):
            subsetDF = gradesDF[(gradesDF['assignment_id']==assignmentID) & (gradesDF['grader_id']==graderID)]
            # display(subsetDF)
            upperX = int(subsetDF['points_possible'].iloc[0])
            minScore = min(resultsDF[(resultsDF['assignment_id']==assignmentID)]['score'])
            lowerX = int(minScore-minScore%2)
            xTickStep = 1 if upperX-lowerX < 10 else 2
            shiftValue = int(subsetDF['points_possible'].iloc[0])*outlierFactor

            g = sns.scatterplot(ax=axes[row,col], data=subsetDF, x='score', y='peerGPT_score', hue='Outlier', hue_order=gradesDF['Outlier'].unique(), palette=sns.color_palette()[:2])
            axes[row,col].plot([lowerX,upperX],[lowerX,upperX], lw=1, color='#313232', linestyle='dashed')
            axes[row,col].plot([lowerX+shiftValue,upperX+shiftValue],[lowerX,upperX], lw=1, color='#aaaaaa', linestyle='dashed')
            axes[row,col].plot([lowerX,upperX],[lowerX+shiftValue,upperX+shiftValue], lw=1, color='#aaaaaa', linestyle='dashed')
            g.set_xlim(lowerX,upperX+0.5)
            g.set_xticks(range(lowerX,upperX+1, xTickStep))
            g.set_ylim(lowerX,upperX+0.5)
            g.set_yticks(range(lowerX,upperX+1, xTickStep))
            # g.set_title(f'Assignment ID: {assignmentID},\nGrader ID: {graderID}')
            # g.set_xlabel('Human Grader Score')
            # g.set_ylabel('peerGPT Score')
            g.set_xlabel('Grader Score',fontsize=8)
            g.set_ylabel('peerGPT Score', fontsize=8, rotation=90)
            g.legend([],[], frameon=False)

    pad = 5
    for ax, col in zip(axes[0], resultsDF['assignment_id'].unique()):
        ax.set_title(f'Assignment ID: {col}')
    for ax, row in zip(axes[:,0], sorted(resultsDF['grader_id'].unique())):
            ax.annotate(f'Grader ID: {row}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)

    # plt.show()
    plt.savefig(os.path.join(chartFolder, 'ScatterPlotSpread.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return True

def getHistogramSpreadByAssgn(resultsDF, chartFolder):
    sns.set_theme(style="darkgrid", palette="dark")
    pad = 5

    saveFolder = os.path.join(chartFolder, 'histplots')
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)

    for col, assignmentID in enumerate(resultsDF['assignment_id'].unique()):

        assgnDF = resultsDF[(resultsDF['assignment_id']==assignmentID)]
        critDataDF = pd.DataFrame()
        for index,row in assgnDF.iterrows():
            criterionData = row['data_peerGPT']
            for col in ['submitter_id', 'assignment_id', 'grader_id']:
                criterionData[col] = row[col]
            critDataDF = pd.concat([critDataDF, criterionData])

        peerGPTGradesDF = critDataDF.copy().drop(['points_grade'], 
                                                axis=1, errors='ignore') 
        peerGPTGradesDF = peerGPTGradesDF.rename(columns={'peerGPT_criterion_score':'points_grade'})
        peerGPTGradesDF['Grader Type'] = 'peerGPT'

        gradersDF = critDataDF.copy().drop(['peerGPT_criterion_score'], 
                                                    axis=1, errors='ignore') 
        gradersDF['Grader Type'] = gradersDF['grader_id'].apply(lambda id: f'Grader ID: {id}')

        allCritDF = pd.concat([gradersDF, peerGPTGradesDF])

        fig, axes = plt.subplots(nrows=len(allCritDF['grader_id'].unique()), ncols=len(allCritDF['description_rubric'].unique()), \
                                            figsize=(len(allCritDF['description_rubric'].unique())*3,len(allCritDF['grader_id'].unique())*3), layout="constrained")
        fig.suptitle(f'Assignment ID: {assignmentID}')
        
        for col, descRubric in enumerate(sorted(allCritDF['description_rubric'].unique())):
            for row, graderID in enumerate(sorted(allCritDF['grader_id'].unique())):
                subsetDF = allCritDF[(allCritDF['grader_id']==graderID) & (allCritDF['description_rubric']==descRubric)].fillna(0)
                # display(subsetDF)
                upperX = int(subsetDF['points_rubric'].iloc[0])
                lowerX = int(min(allCritDF[(allCritDF['description_rubric']==descRubric)]['points_grade']))
                xTickStep = 1

                try:
                    g = sns.histplot(ax=axes[row,col], data=subsetDF, x='points_grade', hue='Grader Type', kde=True, multiple="dodge", palette=sns.color_palette()[:2])
                except:
                    g = sns.histplot(ax=axes[row,col], data=subsetDF, x='points_grade', hue='Grader Type', kde=False, multiple="dodge", palette=sns.color_palette()[:2])
                g.set_xlim(lowerX,upperX)
                g.set_xticks(range(lowerX,upperX+1, xTickStep))
                # g.set_title(f'Assignment ID: {assignmentID}')
                # g.set_xlabel('Points Awarded')
                g.set_xlabel('Score Distribution', fontsize=8)
                g.set_ylabel('')
                g.legend([],[], frameon=False)

        for ax, col in zip(axes[0], sorted(allCritDF['description_rubric'].unique())):
            newLine = '\n'.join(textwrap.wrap(col, width=24))
            ax.set_title(f'{newLine}')
        # for ax, row in zip(axes[:,0], sorted(allCritDF['grader_id'].unique())):
        #     ax.set_ylabel(f'Grader ID: {row}', rotation=90)
        for ax, row in zip(axes[:,0], sorted(allCritDF['grader_id'].unique())):
            ax.annotate(f'Grader ID: {row}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)
        # plt.show()
        # break
        plt.savefig(os.path.join(saveFolder, f'{assignmentID}-HistogramPlotSpread.png'), dpi=300, bbox_inches='tight')
        plt.close()
    return True

def getScatterplotSpreadByAssgn(resultsDF, chartFolder, outlierFactor = 0.15):
    sns.set_theme(style="darkgrid", palette="dark")
    pad = 5

    saveFolder = os.path.join(chartFolder, 'scatterplots')
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)

    for col, assignmentID in enumerate(resultsDF['assignment_id'].unique()):
        assgnDF = resultsDF[(resultsDF['assignment_id']==assignmentID)]
        critDataDF = pd.DataFrame()
        for index,row in assgnDF.iterrows():
            criterionData = row['data_peerGPT']
            for col in ['submitter_id', 'assignment_id', 'grader_id']:
                criterionData[col] = row[col]
            critDataDF = pd.concat([critDataDF, criterionData])

        allCritDF = critDataDF.copy()
        allCritDF['Score Difference'] = allCritDF['peerGPT_criterion_score']-allCritDF['points_grade']
        allCritDF['Outlier'] = (allCritDF['Score Difference']/(allCritDF['points_rubric']*outlierFactor)).apply(lambda scoreDiff: 'Outlier' if abs(scoreDiff)>1 else 'In range')

        fig, axes = plt.subplots(nrows=len(allCritDF['grader_id'].unique()), ncols=len(allCritDF['description_rubric'].unique()), \
                                            figsize=(len(allCritDF['description_rubric'].unique())*3,len(allCritDF['grader_id'].unique())*3), layout="constrained")
        fig.suptitle(f'Assignment ID: {assignmentID}')
        
        for col, descRubric in enumerate(sorted(allCritDF['description_rubric'].unique())):
            for row, graderID in enumerate(sorted(allCritDF['grader_id'].unique())):
                subsetDF = allCritDF[(allCritDF['assignment_id']==assignmentID) & (allCritDF['grader_id']==graderID)]
                # display(subsetDF)
                upperX = int(subsetDF['points_rubric'].iloc[0])+1
                lowerX = int(min(allCritDF[(allCritDF['assignment_id']==assignmentID)]['points_grade']))
                xTickStep = 1
                shiftValue = int(subsetDF['points_rubric'].iloc[0])*outlierFactor

                g = sns.scatterplot(ax=axes[row,col], data=subsetDF, x='points_grade', y='peerGPT_criterion_score', hue='Outlier', hue_order=allCritDF['Outlier'].unique(), palette=sns.color_palette()[:2])
                axes[row,col].plot([lowerX,upperX],[lowerX,upperX], lw=1, color='#313232', linestyle='dashed')
                axes[row,col].plot([lowerX+shiftValue,upperX+shiftValue],[lowerX,upperX], lw=1, color='#aaaaaa', linestyle='dashed')
                axes[row,col].plot([lowerX,upperX],[lowerX+shiftValue,upperX+shiftValue], lw=1, color='#aaaaaa', linestyle='dashed')
                g.set_xlim(lowerX-0.1,upperX+0.1)
                g.set_xticks(range(lowerX,upperX+1, xTickStep))
                g.set_ylim(lowerX-0.1,upperX+0.1)
                g.set_yticks(range(lowerX,upperX+1, xTickStep))
                # g.set_title(f'Assignment ID: {assignmentID},\nGrader ID: {graderID}')
                # g.set_xlabel('Human Grader Score')
                # g.set_ylabel('peerGPT Score')
                g.set_xlabel('Grader Score',fontsize=8)
                g.set_ylabel('peerGPT Score', fontsize=8, rotation=90)
                g.legend([],[], frameon=False)

        for ax, col in zip(axes[0], sorted(allCritDF['description_rubric'].unique())):
            newLine = '\n'.join(textwrap.wrap(col, width=24))
            ax.set_title(f'{newLine}')
        # for ax, row in zip(axes[:,0], sorted(allCritDF['grader_id'].unique())):
        #     ax.set_ylabel(, rotation=90)
        for ax, row in zip(axes[:,0], sorted(allCritDF['grader_id'].unique())):
            ax.annotate(f'Grader ID: {row}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)
        # plt.show()
        # break
        plt.savefig(os.path.join(saveFolder, f'{assignmentID}-ScatterPlotSpread.png'), dpi=300, bbox_inches='tight')
        plt.close()
    return True

def saveGraderPeerGPTMeanScoreDiff(resultsDF, saveName, excelFolder):
    excludeDF = resultsDF.copy()
    excludeDF['Score Difference'] = excludeDF['peerGPT_score']-excludeDF['score']

    meanDiffDict = {}
    for group in excludeDF.groupby(['grader_id','assignment_id']):
        if group[0][0] not in meanDiffDict:
            meanDiffDict[group[0][0]] = {}
        meanDiffDict[group[0][0]][group[0][1]] = group[1]["Score Difference"].mean()
    meanDiffDF = pd.DataFrame(meanDiffDict)
    meanDiffDF.to_excel(os.path.join(excelFolder, saveName+' - Grader - peerGPT Score Difference.xlsx'))
    return meanDiffDF

def buildFullInfoDF(config, resultsDF, saveName, excelFolder):
    critDataDF = pd.DataFrame()
    for index,row in resultsDF.iterrows():
        criterionData = row['data_peerGPT']
        for col in ['submitter_id', 'assignment_id', 'grader_id']:
            criterionData[col] = row[col]
        critDataDF = pd.concat([critDataDF, criterionData])
    allCritDF = critDataDF.drop(['mastery_points','ignore_for_scoring','title','peerGPT_criterion_id','description_grade'],
                            axis=1, errors='ignore')

    meanInfoList = []
    for group in allCritDF.groupby(['assignment_id','criterion_id','grader_id']):
        meanInfoList.append({'assignment_id':group[0][0], 'criterion_id':group[0][1], 'grader_id':group[0][2], \
                            'Grader Mean':group[1]['points_grade'].mean(), \
                            'Grader Std. Dev.':group[1]['points_grade'].std(), \
                            'peerGPT Mean':group[1]['peerGPT_criterion_score'].mean(), \
                            'peerGPT Std. Dev.':group[1]['peerGPT_criterion_score'].std(), \
                            'Correlation Score':group[1]['peerGPT_criterion_score'].corr(group[1]['points_grade'])})
    meanInfoDF = pd.DataFrame(meanInfoList)
    meanInfoDF['Mean Difference'] = meanInfoDF['peerGPT Mean'] - meanInfoDF['Grader Mean']

    assignmentDF = getGRAData(config)[['assignment_id', 'assignment_title']].drop_duplicates()
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

    fullInfoDF.to_excel(os.path.join(excelFolder, saveName+' - Grader Difference Table.xlsx'))

    rubricInfo = getGRAData(config)[['assignment_id', 'data_rubric']].drop_duplicates('assignment_id').reset_index(drop=True)
    rubricOrderDict = {}
    for index, row in rubricInfo.iterrows():
        rubricOrderDict[row['assignment_id']] = pd.DataFrame(row['data_rubric'])['description'].tolist()

    return fullInfoDF, rubricOrderDict

def getMeanDiffCharts(fullInfoDF, rubricOrderDict, chartFolder):
    sns.set_theme(style="darkgrid") #, palette="dark")
    saveMeanFolder = os.path.join(chartFolder, 'Mean Difference Charts')
    if not os.path.exists(saveMeanFolder):
        os.mkdir(saveMeanFolder)

    for AID in fullInfoDF['assignment_id'].unique():
        subsetDF =  fullInfoDF[fullInfoDF['assignment_id']==AID]
        plt.clf()

        graderCount = len(subsetDF['grader_id'].unique())

        upperY = math.ceil(max(subsetDF['Mean Difference'])*10)/10
        lowerY = math.floor(min(subsetDF['Mean Difference'])*10)/10
        if upperY<0:
            upperY = 0
        if upperY-lowerY>1.6:
            tickStep = 0.2
        else:
            tickStep = 0.1
        if lowerY<0 and upperY>0:
            tickSpace = np.concatenate((np.arange(lowerY-lowerY%tickStep,0, tickStep), \
                                        np.arange(0,upperY+upperY%tickStep+0.1, tickStep)))
        else:
            tickSpace = np.arange(lowerY,upperY+0.1, tickStep)
        # sns.set(rc={'figure.figsize':((3/2)*len(rubricOrderDict[AID]),3)})

        g = sns.stripplot(data=subsetDF, x='description_rubric', y='Mean Difference', \
                            order = rubricOrderDict[AID], \
                            hue='grader_id', dodge=False, jitter=True, \
                            palette=sns.color_palette(n_colors=graderCount)[:graderCount])
        
        plt.axhline(y=0, color='#313232', linestyle='--')

        g.set_ylim(lowerY-0.05,upperY+0.05)
        g.set_yticks(tickSpace)
        g.set_xlabel('Criteria', fontsize=12)
        g.set_ylabel('Mean Difference', fontsize=12, rotation=90)

        g.set_xticks(g.get_xticks())

        wrapSize = 14 if len(rubricOrderDict[AID]) < 6 else 12 
        g.set_xticklabels([textwrap.fill(t.get_text(), wrapSize, break_long_words=False) \
                                         for t in g.get_xticklabels()], size=9)

        g.set_title(textwrap.fill(subsetDF['assignment_title'].iloc[0], 50))

        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, title='Grader ID')
        # plt.show()
        g.get_figure().savefig(os.path.join(saveMeanFolder, f'{AID}-MeanDiffSpread.png'), \
                               dpi=300, bbox_inches='tight')
        plt.close()
    return True

def getMeanDiffPercentCharts(fullInfoDF, rubricOrderDict, chartFolder):
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
        g.get_figure().savefig(os.path.join(saveMeanFolder, f'{AID}-MeanDiffSpread.png'), dpi=300, bbox_inches='tight')
        plt.close()
    return True

def confindenceInterval(data, confidence=0.9):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

def getCIOutlierGraderDetails(fullInfoDF, saveName, excelFolder, confidenceLevel=0.93):
    confidenceList = np.arange(0.85,1, 0.01)
    confidenceDataDict = {}

    for confidence in confidenceList:
        outsideCIDF = pd.DataFrame()
        for AID in fullInfoDF['assignment_id'].unique():
            for CID in fullInfoDF[(fullInfoDF['assignment_id']==AID)]['criterion_id'].unique(): 
                subsetDF =  fullInfoDF[(fullInfoDF['assignment_id']==AID) & (fullInfoDF['criterion_id']==CID)]
                meanDiffDict = dict(zip(subsetDF['grader_id'].tolist(),subsetDF['Mean Difference %'].tolist()))
                lower,upper = confindenceInterval(list(meanDiffDict.values()), confidence)
                for grader in meanDiffDict:
                    if meanDiffDict[grader]<lower or meanDiffDict[grader]>upper:
                        # display(subsetDF[subsetDF['grader_id']==grader])
                        outsideCIDF = pd.concat([outsideCIDF,subsetDF[subsetDF['grader_id']==grader]])

        if np.round(confidence,4)==confidenceLevel:
            # display(outsideCIDF)
            outsideCIDF.to_excel(os.path.join(excelFolder, saveName+f' - Outliers at {confidenceLevel} Conf. Level.xlsx'))
            retrievedCIDF = outsideCIDF.copy()
        if not outsideCIDF.empty:
            confidenceDataDict[np.round(confidence, 2)] = outsideCIDF['grader_id'].value_counts().to_dict()
        else:
            break

    meanConDF = pd.DataFrame(confidenceDataDict).sort_index()
    meanConDF.to_excel(os.path.join(excelFolder, saveName+' - Confidence in Mean Diff % Table.xlsx'))

    return meanConDF, retrievedCIDF

def getHighErrorCriteria(config, fullInfoDF, saveName, excelFolder, scoreThreshold = 10):
    pd.set_option('display.precision', 2)

    descDataDF = config.critDescDF[['custom_description', 'assignment_id', 'id']]
    criteriaIssuesDF = pd.DataFrame()
    for AID in fullInfoDF['assignment_id'].unique():
        for CID in fullInfoDF[(fullInfoDF['assignment_id']==AID)]['criterion_id'].unique(): 
            subsetDF =  fullInfoDF[(fullInfoDF['assignment_id']==AID) \
                                   & (fullInfoDF['criterion_id']==CID)]
            if subsetDF['Mean Difference %'].mean() > scoreThreshold:
                issueDF = subsetDF[['assignment_id', 'criterion_id', \
                                    'assignment_title', 'description_rubric', \
                                    'points_rubric', 'All Graders Mean', \
                                    'Global peerGPT Mean']].drop_duplicates()
                issueDF['Mean Difference %'] = subsetDF['Mean Difference %'].mean()
                criteriaIssuesDF = pd.concat([criteriaIssuesDF, issueDF])

    criteriaIssuesDF = criteriaIssuesDF.merge(descDataDF, left_on=['assignment_id', 'criterion_id'], \
                                            right_on=['assignment_id', 'id'])
    del criteriaIssuesDF['id']
    criteriaIssuesDF = criteriaIssuesDF.rename(columns={'assignment_title':'Title', \
                                                        'custom_description':'Supplemental Description', \
                                                        'description_rubric':'Rubric', \
                                                        'points_rubric':'Max Score'}).reset_index(drop=True)
    criteriaIssuesDF.to_excel(os.path.join(excelFolder, saveName+' - High Error Criteria.xlsx'))

    return criteriaIssuesDF


def setUpPostProcessVars(versionControl='V3', promptVersion='P2', simpleCourseName='MOVESCI'):
    varsDict = {varName:None for varName in ['config', 'saveName', \
                                             'chartFolder', 'excelFolder', \
                                             'resultsDF', 'errorDF', \
                                             'fullInfoDF', 'rubricOrderDict']}

    varsDict['config'] = Config()
    varsDict['config'].setFromEnv()
    varsDict['config'].simpleCourseName = simpleCourseName
    varsDict['saveName'] = f"{varsDict['config'].simpleCourseName}-{versionControl}-{promptVersion}"

    varsDict['config'].setSaveDetails(varsDict['saveName'])

    varsDict['chartFolder'] = os.path.join(varsDict['config'].baseOutputFolder, \
                               varsDict['config'].outputFolders['CHART_OUTPUT'], \
                               varsDict['config'].fullName)

    varsDict['excelFolder'] = os.path.join(varsDict['config'].baseOutputFolder, \
                               varsDict['config'].outputFolders['EXCEL_OUTPUT'], \
                               varsDict['config'].fullName)
    
    varsDict['resultsDF'] = convertPicklesToDF('saves', varsDict['config'])
    varsDict['errorDF'] = convertPicklesToDF('errors', varsDict['config'])
    varsDict['fullInfoDF'], varsDict['rubricOrderDict'] = buildFullInfoDF(varsDict['config'], \
                                                                          varsDict['resultsDF'], \
                                                                          varsDict['saveName'], \
                                                                          varsDict['excelFolder'])
    
    return varsDict


def buildComparerFullInfoDF(config, compareResultsDF, saveName, excelFolder):
    critDataDF = pd.DataFrame()
    for index,row in compareResultsDF.iterrows():
        criterionData_P1 = row['data_peerGPT_P1']
        criterionData_P2 = row['data_peerGPT_P2']

        mergedCriterionData = criterionData_P1.merge(criterionData_P2, on=criterionCommonCols, \
                                                    how='inner', suffixes=('_P1', '_P2'))
        critDataDF = pd.concat([critDataDF, mergedCriterionData])
    
    for dataCol in critDataDF:
        for colName in ['mastery_points', 'ignore_for_scoring', 'title', \
                        'peerGPT_criterion_id', 'description_grade']:
            if dataCol.startswith(colName):
                critDataDF = critDataDF.drop(columns=[dataCol])

    meanInfoList = []
    for group in critDataDF.groupby(['assignment_id','criterion_id','grader_id']):
        meanInfoList.append({'assignment_id':group[0][0], 'criterion_id':group[0][1], 'grader_id':group[0][2], \
                            'Grader Mean':group[1]['points_grade'].mean(), \
                            'Grader Std. Dev.':group[1]['points_grade'].std(), \
                            'peerGPT Mean P1':group[1]['peerGPT_criterion_score_P1'].mean(), \
                            'peerGPT P1 Std. Dev. P1':group[1]['peerGPT_criterion_score_P1'].std(), \
                            'Correlation Score P2':group[1]['peerGPT_criterion_score_P1'].corr(group[1]['points_grade']), \
                            'peerGPT Mean P2':group[1]['peerGPT_criterion_score_P2'].mean(), \
                            'peerGPT Std. Dev. P2':group[1]['peerGPT_criterion_score_P2'].std(), \
                            'Correlation Score P2':group[1]['peerGPT_criterion_score_P2'].corr(group[1]['points_grade'])})
    meanInfoDF = pd.DataFrame(meanInfoList)
    meanInfoDF['Mean Difference P1'] = meanInfoDF['peerGPT Mean P1'] - meanInfoDF['Grader Mean']
    meanInfoDF['Mean Difference P2'] = meanInfoDF['peerGPT Mean P2'] - meanInfoDF['Grader Mean']

    assignmentDF = getGRAData(config)[['assignment_id', 'assignment_title']].drop_duplicates()
    baseInfoDF = critDataDF[['assignment_id', 'criterion_id', 'description_rubric', 'points_rubric']].drop_duplicates()
    baseInfoDF = baseInfoDF.merge(assignmentDF, on='assignment_id')

    globalMeanList = [{'assignment_id':group[0][0], 'criterion_id':group[0][1], \
                    'All Graders Mean':group[1]['points_grade'].mean(), \
                    'All Graders Std. Dev.':group[1]['points_grade'].std(), \
                    'Global peerGPT Mean P1':group[1]['peerGPT_criterion_score_P1'].mean(), \
                    'Global peerGPT Std. Dev. P1':group[1]['peerGPT_criterion_score_P1'].std(), \
                    'Global peerGPT Mean P2':group[1]['peerGPT_criterion_score_P2'].mean(), \
                    'Global peerGPT Std. Dev. P2':group[1]['peerGPT_criterion_score_P2'].std()} \
                        for group in critDataDF.groupby(['assignment_id', 'criterion_id'])]
    globalMeanDF = pd.DataFrame(globalMeanList)
    baseInfoDF = baseInfoDF.merge(globalMeanDF, on=['assignment_id', 'criterion_id'])

    fullInfoDF = meanInfoDF.merge(baseInfoDF, on=['assignment_id', 'criterion_id'])
    fullInfoDF['Mean Difference % P1'] = 100*fullInfoDF['Mean Difference P1'].div(fullInfoDF['points_rubric'])
    fullInfoDF['Mean Difference % P2'] = 100*fullInfoDF['Mean Difference P2'].div(fullInfoDF['points_rubric'])

    fullInfoDF.to_excel(os.path.join(excelFolder, saveName+' - Grader Difference Table.xlsx'))

    rubricInfo = getGRAData(config)[['assignment_id', 'data_rubric']]\
                    .drop_duplicates('assignment_id')\
                    .reset_index(drop=True)
    rubricOrderDict = {}
    for index, row in rubricInfo.iterrows():
        rubricOrderDict[row['assignment_id']] = pd.DataFrame(row['data_rubric'])['description'].tolist()

    fullInfoDF.to_excel(os.path.join(excelFolder, saveName+' - Grader Difference Table.xlsx'))
    return fullInfoDF, rubricOrderDict
