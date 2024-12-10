# **Fine-Tuned Llama Model for Efficient Inference with Gradio**

This project demonstrates how to fine-tune a large language model using Parameter-Efficient Fine-Tuning (PEFT) and integrates it into a chatbot application using Gradio. The goal is to make inference on large language models more efficient and accessible, even with limited GPU resources.

##Overview

This project focuses on fine-tuning the Unsloth/Llama-3.2-1B-Instruct model using LoRA (Low-Rank Adaptation), a PEFT technique that reduces the memory requirements of large models. We use 4-bit precision to further optimize memory consumption, making it feasible to run on lower-end GPUs.

Once the model is fine-tuned, a Gradio-powered chatbot interface is provided to interact with the model. The chatbot can engage in conversations and respond to user queries based on the training it received.
Key Features

Efficient Fine-Tuning: We fine-tuned the 1B parameter Llama model using LoRA in just 2200 steps with the FineTome-100k dataset, significantly reducing memory usage while maintaining quality.

Real-Time Chatbot: The fine-tuned model powers an interactive chatbot built with Gradio, where users can send messages and receive dynamic, context-aware responses.

Streaming Responses: Chatbot responses are generated in real time for a more interactive experience, providing a natural flow of conversation.

##How It Works
Fine-Tuning the Model

The model was fine-tuned using a subset of the FineTome-100k dataset, which has been pre-processed into the HuggingFace format. The 4-bit precision ensures that the model can run on devices with limited GPU memory. To preserve progress, automatic checkpointing was used, ensuring that every 50 steps the model’s state was saved.
Chatbot Application

Once the model is fine-tuned, a Gradio interface is built for interacting with the model. The chatbot can hold a conversation, remember previous interactions, and provide dynamic answers. You can run this app either locally or on HuggingFace Spaces.
Setup

To get started with the project, follow the steps below:
Clone this repository:

git clone https://github.com/manjy0t/lab2_scalable.git
cd llama-chatbot

Install dependencies:

You'll need Python 3.8+ and some dependencies to run the application. Install them using pip:

pip install -r requirements.txt

Run the chatbot locally:

Once dependencies are installed, you can run the Gradio interface with the following command:
python app.py
This will start a local server, and you can interact with the chatbot via a web interface.
Deploy on HuggingFace Spaces (optional):
If you wish to deploy the chatbot on HuggingFace Spaces for public access, you can upload the model to HuggingFace and follow the instructions on their site for setting up a space.

File Structure
app.py: The main Python script that initializes the Gradio app and handles chatbot interactions.
chatbot.py: Contains the logic for the chatbot, including handling user input and maintaining conversation history.
requirements.txt: A list of dependencies required to run the application.
fine_tuning_notebook.ipynb: A Jupyter notebook that details the fine-tuning process using the LoRA technique.

Inference Pipeline

The fine-tuned model is loaded and used for inference through the following steps:

Tokenization: The user’s input is tokenized using the AutoTokenizer from HuggingFace.
Model Generation: The tokenized input is passed through the fine-tuned Llama-3.2-1B-Instruct model to generate a response. The model uses LoRA to efficiently handle this in reduced memory.
Streaming Output: The generated response is decoded and sent back to the user in real-time, allowing for smooth interactions.

Requirements

To run this project, you'll need the following Python packages:

transformers==4.29.0
torch==2.0.1
gradio==3.28.0

You can install them all at once using:

pip install -r requirements.txt



