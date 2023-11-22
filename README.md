## Introduction
------------
The BraveneWWWorld Chat App is a Python application that allows you to chat with various documents in the context of intimacy and its many nuances that deserve individual exploration. You can ask questions about the context of documents using natural language, and the app will provide relevant answers based on the content of the documents. This app uses a language model to generate accurate answers to your questions using informal and formal language. Please note that the application will respond to queries based on the documents uploaded and also based on information coming from the World Wide Web Cloud.

This app is built for contemporany art projet. braveneWWWorld” is an artistic in(ter)disciplinary group founded by Contemporary Arts Performer, storyteller and community organizer Brandy Butler (US/CH), together with artist, programmer and filmmaker Juan Ferrari (UY/CH/FR). Both artists met online through a collaborative project at Theater Neumarkt in the first quarantine of 2020, and connected on their joint interest in technology. They immediately shared a desire to explore the meaning that human subjectivity gives to technology and the relationship it creates between them.

## How It Works
------------

![BraveneWWWorld Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. The app reads multiple documents from a diretory and extracts their text content.

2. Document Types: App alows .PDF, .DOC*, .CSV, .TXT and .SRT document types.

3. Document Chunking: The extracted document content is divided into smaller chunks that can be processed effectively.

4. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the content chunks.

5. Similarity Matching: When you ask a question, the app compares it with the content chunks and identifies the most semantically similar ones.

6. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the documents.

## Dependencies and Installation
----------------------------
To install the BraveneWWWorld Chat App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip3 install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the BraveneWWWorld Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple documents into the application by placing them in the "documents/context" directory and press the "PROCESS DADA" button.

5. Ask questions in natural language about the loaded documents using the chat interface.

## Contributing
------------
This repository is intended for art project purposes and does not accept further contributions.
