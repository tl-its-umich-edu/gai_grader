from helper import *
from multiprocessing import Pool

# There some issues with multiprocessing that causes this to run erratically.

def main():
    config = Config()
    config.setFromEnv()

    versionControl = 'V3'
    promptVersion='P2'
    fullName = f'{versionControl}-{promptVersion}'
    config.setSaveDetails(fullName)
    config.saveTemplatePrompt()

    config.overWriteSave = False
    config.customDescMode = False

    config.poolSize = 16

    rowDataList = getGRAData(config, mpMode=True)

    pool = Pool(config.poolSize)
    pool.imap(checkRunSaveMP, rowDataList)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()