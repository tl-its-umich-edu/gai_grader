from helper import *

from tqdm import tqdm
from multiprocessing import Pool

# There some issues with multiprocessing that causes this to run erratically.

def main():
    config = Config()
    config.setFromEnv()

    versionControl = 'V1'
    promptVersion='P2'
    fullName = f'{versionControl}-{promptVersion}'
    config.setSaveDetails(fullName)
    config.saveTemplatePrompt()

    config.overWriteSave = False

    config.simpleCourseName = 'Movement Science'
    config.poolSize = 8

    rowDataList = getGRAData(config, mpMode=True)

    pool = Pool(config.poolSize)
    results = tqdm(pool.imap(checkRunSaveMP, rowDataList), total=len(rowDataList))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()