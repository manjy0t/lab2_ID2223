# **Fine-Tuned Llama Model for Efficient Inference with Gradio**

This project demonstrates how to fine-tune a large language model using Parameter-Efficient Fine-Tuning (PEFT) and integrates it into a chatbot application using Gradio. The goal is to make inference on large language models more efficient and accessible, even with limited GPU resources.

## Overview

This project focuses on fine-tuning the Unsloth/Llama-3.2-1B-Instruct model using LoRA (Low-Rank Adaptation), a PEFT technique that reduces the memory requirements of large models. We use 4-bit precision to further optimize memory consumption, making it feasible to run on lower-end GPUs.

Once the model is fine-tuned, a Gradio-powered chatbot interface is provided to interact with the model. The chatbot can engage in conversations and respond to user queries based on the training it received.
Key Features

Efficient Fine-Tuning: We fine-tuned the 1B parameter Llama model using LoRA in just 4000 steps with the FineTome-100k dataset, significantly reducing memory usage while maintaining quality.

Real-Time Chatbot: The fine-tuned model powers an interactive chatbot built with Gradio, where users can send messages and receive dynamic, context-aware responses.

Streaming Responses: Chatbot responses are generated in real time for a more interactive experience, providing a natural flow of conversation.

## How It Works
Fine-Tuning the Model

The model was fine-tuned using a subset of the FineTome-100k dataset, which has been pre-processed into the HuggingFace format. The 4-bit precision ensures that the model can run on devices with limited GPU memory. To preserve progress, automatic checkpointing was used, ensuring that every 50 steps the model’s state was saved.
Chatbot Application

Once the model is fine-tuned, a Gradio interface is built for interacting with the model. The chatbot can hold a conversation, remember previous interactions, and provide dynamic answers. You can run this app either locally or on HuggingFace Spaces.
Setup

To get started with the project, follow the steps below:
1. Clone this repository:
git clone https://github.com/manjy0t/lab2_scalable.git

cd llama-chatbot

2. Install dependencies:

You'll need Python 3.8+ and some dependencies to run the application. Install them using pip:

pip install -r requirements.txt

3. Run the chatbot locally:

Once dependencies are installed, you can run the Gradio interface with the following command:
python app.py
This will start a local server, and you can interact with the chatbot via a web interface.
Deploy on HuggingFace Spaces (optional):
If you wish to deploy the chatbot on HuggingFace Spaces for public access, you can upload the model to HuggingFace and follow the instructions on their site for setting up a space.

File Structure
app.py: The main Python script that initializes the Gradio app and handles chatbot interactions.
requirements.txt: A list of dependencies required to run the application.

Inference Pipeline

The fine-tuned model is loaded and used for inference through the following steps:

Tokenization: The user’s input is tokenized using the AutoTokenizer from HuggingFace.
Model Generation: The tokenized input is passed through the fine-tuned Llama-3.2-1B-Instruct model to generate a response. The model uses LoRA to efficiently handle this in reduced memory.
Streaming Output: The generated response is decoded and sent back to the user in real-time, allowing for smooth interactions.

## Requirements

To run this project, you'll need the following Python packages:

transformers==4.29.0
torch==2.0.1
gradio==3.28.0

You can install them all at once using:

pip install -r requirements.txt

## Finetuning the models
1. Model-Centric Approach

The model-centric approach primarily focuses on improving the model architecture, training algorithms, and hyperparameters. It emphasizes optimizing the learning process by tweaking aspects that directly impact the model’s learning ability and how it generalizes from training data. We focussed on fine-tuning the model’s hyperparameters, such as learning rate, batch size, and weight decay. These hyperparameters control how the model optimizes during training, affecting both convergence speed and training stability.

2. Data-Centric Approach

The data-centric approach, on the other hand, emphasizes improving the quality of the data rather than changing the model itself. This approach assumes that high-quality data is a critical factor in model performance and that improving the dataset often yields greater benefits than further model refinement. So, we may strategically select a smaller, but more representative subset of the data, to help the model focus on high-quality examples rather than noisy ones.
If certain classes are underrepresented in the dataset, we can over-sample them or use other techniques like Random oversampling, SMOTE (Synthetic Minority Over-sampling Technique), AdaSyn, Borderline SMOTE to balance class distributions.



