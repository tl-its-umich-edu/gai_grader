from helper import *
from multiprocessing import Pool

# Set poolsize here manually. Keep it less than 16 to avoid overloading API endpoint.

def main():
    config = Config()
    config.setFromEnv()

    # Option to use custom variables here.
    # versionControl = 'V3'
    # promptVersion = 'P4'
    # courseShorthand = 'ECON'

    # customConfigParams = {
    #                 'Save Name':f'{courseShorthand}-{versionControl}-{promptVersion}', 
    #                 'Overwrite Saves': False, 
    #                 'Use Custom Desc.':False
    #                 }
    # config.setSaveDetails(customConfigParams)

    config.setSaveDetails()
    config.saveTemplatePrompt()

    config.poolSize = 16

    rowDataList = getGRAData(config, mpMode=True)

    pool = Pool(config.poolSize)
    pool.imap(checkRunSaveMP, rowDataList)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()