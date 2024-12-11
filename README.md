# **Fine-Tuned Llama Model for Efficient Inference with Gradio**

This project demonstrates how to fine-tune a large language model using Parameter-Efficient Fine-Tuning (PEFT) and integrates it into a chatbot application using Gradio. The goal is to make inference on large language models more efficient and accessible, even with limited GPU resources.

## Overview

This project focuses on fine-tuning the Unsloth/Llama-3.2-1B-Instruct model using LoRA (Low-Rank Adaptation), a PEFT technique that reduces the memory requirements of large models. We use 4-bit precision to further optimize memory consumption, making it feasible to run on lower-end GPUs.

Once the model is fine-tuned, a Gradio-powered chatbot interface is provided to interact with the model. The chatbot can engage in conversations and respond to user queries based on the training it received.

There are three bots using this fine-tuned inference:
1. Rudimentary bot: It can respond to general user-queries concerning different topics.
2. Image-description bot: You upload an image and the bot would describe it in a creative way.
3. DnD inspied bot: The bot will act as the story-teller and would guide the player when supplied with prompts. It keeps the context and previous user decisions in mind. 

Key Features

Efficient Fine-Tuning: We fine-tuned the 1B parameter Llama model using LoRA in just 4000 steps with the FineTome-100k dataset, significantly reducing memory usage while maintaining quality.

Real-Time Chatbot: The fine-tuned model powers an interactive chatbot built with Gradio, where users can send messages and receive dynamic, context-aware responses.

Streaming Responses: Chatbot responses are generated in real time for a more interactive experience, providing a natural flow of conversation.

## Requirements

To run this project, you'll need the following Python packages:

transformers==4.29.0
torch==2.0.1
gradio==3.28.0

You can install them all at once using:

pip install -r requirements.txt

## How It Works
Fine-Tuning the Model

The model was fine-tuned using unsloth on a subset of the FineTome-100k dataset, which has been pre-processed into the HuggingFace format. The 4-bit precision ensures that the model can run on devices with limited GPU memory. To preserve progress, automatic checkpointing was used, ensuring that every 50 steps the model’s state was saved.

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



## Finetuning the models
1. Model-Centric Approach

The model-centric approach primarily focuses on improving the model architecture, training algorithms, and hyperparameters. It emphasizes optimizing the learning process by tweaking aspects that directly impact the model’s learning ability and how it generalizes from training data. We focussed on fine-tuning the model’s hyperparameters, such as learning rate, batch size, and weight decay. These hyperparameters control how the model optimizes during training, affecting both convergence speed and training stability. We experimented with the parameters lr_schedulers(linear, cosine, constant), weight_decays( 0.0, 0.01,0.1), and batch_sizes(2,4,8,16).

![Screenshot 2024-12-11 075626](https://github.com/user-attachments/assets/a1e9fae1-a4fa-4a9f-9e24-82b1fa4341f3)

![Screenshot 2024-12-10 184407](https://github.com/user-attachments/assets/fb5c7b42-00f7-462b-b69c-13e21e752bea)

![Screenshot 2024-12-10 184514](https://github.com/user-attachments/assets/4a63e8fc-bdb6-4fb0-bcfa-0a7a4d4985fc)

Further, the hyperparameters for the LoRA adapter could also be tested. This was not implemented due to time contraints. Reasonably, the rank of LoRA matrices could impact both loss and training time, as higher rank allows the model to capture more information because it adds more trainable parameters. This might reduce the loss for complex datasets. However, setting the rank too high could lead to overfitting. Increasing the rank increases the number of parameters to optimize, which leads to longer training times. Similarly, the scaling factor (alpha) could affect training time as larger alpha may require more gradient updates to stabilize the model, slightly increasing training time due to smaller learning steps required for convergence. Higher values of alpha also amplify the updates, which might improve loss convergence initially but could lead to instability if the updates are too large. Another factor that could influence these performance metrics is dropout. Dropout helps prevent overfitting by randomly disabling connections during training. Higher dropout rates might increase the loss initially due to reduced capacity but can result in better generalization. As for training time, the effect is less significant. Applying dropout slightly increases training time since the effective model capacity is reduced, requiring some more updates to converge.

2. Data-Centric Approach

The data-centric approach, on the other hand, emphasizes improving the quality of the data rather than changing the model itself. This approach assumes that high-quality data is a critical factor in model performance and that improving the dataset often yields greater benefits than further model refinement. So, we may strategically select a smaller, but more representative subset of the data, to help the model focus on high-quality examples rather than noisy ones.
If certain classes are underrepresented in the dataset, we can over-sample them or use other techniques like Random oversampling, SMOTE (Synthetic Minority Over-sampling Technique), AdaSyn, Borderline SMOTE to balance class distributions.



