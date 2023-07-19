from helper import *
from multiprocessing import Pool

# There some issues with multiprocessing that causes this to run erratically.

def main():
    config = Config()
    config.setFromEnv()

    versionControl = 'V3'
    promptVersion='P3'
    simpleCourseName = 'ECON'
    fullName = f'{simpleCourseName}-{versionControl}-{promptVersion}'
    
    config.overWriteSave = False
    config.customDescMode = False

    config.setSaveDetails(fullName)
    config.saveTemplatePrompt()

    config.poolSize = 16

    rowDataList = getGRAData(config, mpMode=True)

    # for rowData in rowDataList:
    #     checkRunSaveMP(rowData)
    pool = Pool(config.poolSize)
    pool.imap(checkRunSaveMP, rowDataList)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()