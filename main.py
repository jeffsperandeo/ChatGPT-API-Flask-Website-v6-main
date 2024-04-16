# pip install Flask openai
# openai migrate

with open("./static/chromosomes.py") as f:
    exec(f.read())

from flask import Flask, request, render_template, redirect
import openai
# Import the os module to interact with operating system features.  This includes fetching environment variables.
import os
# Import specific classes or functions directly from their modules to avoid prefixing them with the module name.
# Import the OpenAI library
import openai
# from openai import OpenAI
# Import the load_dotenv and find_dotenv functions from the dotenv package.
# These are used for loading environment variables from a .env file.
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file.
_ = load_dotenv(find_dotenv())

# Set the OpenAI API key by retrieving it from the environment variables.
# OpenAI.api_key = os.environ['OPENAI_API_KEY']
# OpenAI.assistant_key = os.environ['ASSISTANT_ID']

app = Flask(__name__)

# can be empty
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key = os.environ.get("OPENAI_API_KEY"),
    # assistant_id = os.environ.get('ASSISTANT_ID')
)

# https://stackoverflow.com/questions/78018805/how-to-execute-custom-actions-with-chatgpt-assistants-api
assistantId = os.environ.get('ASSISTANT_ID')

# https://www.youtube.com/watch?v=pZUDEQs89zc
# https://mer.vin/2023/11/chatgpt-assistants-api/
import time

# Step 1: Create an Assistant
##assistant = client.beta.assistants.create(
##    name="Transcriptome Classifier",
##    instructions="I want you to act as a scientific data visualizer. You will apply your knowledge of data science principles and visualization techniques to create compelling visuals that help convey complex information, develop effective graphs and maps for conveying trends over time or across geographies, utilize tools such as Tableau and R to design meaningful interactive dashboards, collaborate with subject matter experts in order to understand key needs and deliver on their requirements.",
##    # model="gpt-4-1106-preview"
##    model="gpt-3.5-turbo"
##)

# Step 2: Create a Thread
thread = client.beta.threads.create()

# Step 3: Add a Message to a Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    ## content = "What information would you like about the transcriptome?",
    content = "How many circRNAs are in hsa_hg38_circRNA.bed?"
)

# Step 4: Run the Assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistantId,
    instructions="Please address the user as Jane Doe. The user has a premium account."
)

print(run.model_dump_json(indent=4))


while True:
    # Wait for 5 seconds
    time.sleep(5)  

    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(run_status.model_dump_json(indent=4))

    # If run is completed, get messages
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        # Loop through messages and print content based on role
        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")
        
        break


## render_template('index.html', prompt=content, response=messages)

## if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(debug=True, host='127.0.0.1', port=5000)
    ## app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=True, host='8080', port=5000)
    # server.run(debug=True, host='0.0.0.0', port=5000)

# https://stackoverflow.com/questions/78018805/how-to-execute-custom-actions-with-chatgpt-assistants-api
# https://platform.openai.com/docs/api-reference/models/delete
# https://www.youtube.com/watch?v=pZUDEQs89zc
# https://platform.openai.com/docs/assistants/tools/knowledge-retrieval