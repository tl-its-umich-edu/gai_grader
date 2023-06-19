# peerGPT Grader - - README.md

Leverage ChatGPT to assist in grading courses.

1. Get OpenAI key for accessing API:
(This is only if you are using a personal account.)
   > Using OpenAI's Web UI, click on "Profile Picture" (Top right of screen) > "View API keys" (https://platform.openai.com/account/api-keys)  
   > Make a new key, and copy the key for use in accessing the API.  

2. Use the `.env.sample` in the config folder to create an `.env` file in the root of the directory. 
   > Specify the API key for OpenAI using the key you copied in step 2.
   <!-- > Specify the settings for DB key using values from the `database-research_ro.json` file. -->
Note: You need to have these keys, there are no default values.

3. Adjust variables in the `.env` file before running if not applying defaults.