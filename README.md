
---

# **Fine-Tuned Llama Model for Efficient Inference with Gradio**  

This project demonstrates how to fine-tune a large language model using Parameter-Efficient Fine-Tuning (PEFT) and integrate it into a chatbot application using Gradio. The goal is to make inference on large language models more efficient and accessible, even with limited GPU resources.  

## **Overview**  

This project focuses on fine-tuning the `Unsloth/Llama-3.2-1B-Instruct` model using LoRA (Low-Rank Adaptation), a PEFT technique that reduces the memory requirements of large models. We use 4-bit precision to further optimize memory consumption, making it feasible to run on lower-end GPUs.  

Once the model is fine-tuned, a Gradio-powered chatbot interface is provided to interact with the model. The chatbot can engage in conversations and respond to user queries based on the training it received.  

There are three bots using this fine-tuned model for inference:  
1. **Rudimentary Bot**: Responds to general user queries on various topics.  
2. **Image-Description Bot**: Allows users to upload an image, which the bot describes creatively.  
3. **DnD-Inspired Bot**: Acts as a storyteller, guiding players through prompts while keeping context and previous user decisions in mind.  

### **Key Features**  
- **Efficient Fine-Tuning**: We fine-tuned the 1B-parameter Llama model using LoRA in just 4,000 steps with the FineTome-100k dataset, significantly reducing memory usage while maintaining quality.  
- **Real-Time Chatbot**: The fine-tuned model powers an interactive chatbot built with Gradio, where users can send messages and receive dynamic, context-aware responses.  
- **Streaming Responses**: Chatbot responses are generated in real time for a more interactive experience, providing a natural flow of conversation.  

---

## **Requirements**  

To run this project, you'll need the following Python packages:  

- `transformers==4.29.0`  
- `torch==2.0.1`  
- `gradio==3.28.0`  

You can install them all at once using:  
```bash
pip install -r requirements.txt
```

---

## **How It Works**  

### **Fine-Tuning the Model**  
The model was fine-tuned using `unsloth` on a subset of the FineTome-100k dataset, preprocessed into HuggingFace format. The 4-bit precision ensures the model can run on devices with limited GPU memory. To preserve progress, automatic checkpointing was used, saving the model’s state every 50 steps.  

### **Chatbot Application**  
Once fine-tuned, a Gradio interface was built to interact with the model. The chatbot can hold conversations, remember previous interactions, and provide dynamic answers.  

#### **Setup**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/manjy0t/lab2_scalable.git  
   cd llama-chatbot  
   ```  

2. Install dependencies:  
   Ensure you have Python 3.8+ installed. Install required packages:  
   ```bash
   pip install -r requirements.txt  
   ```  

3. Run the chatbot locally:  
   After installing dependencies, launch the Gradio interface:  
   ```bash
   python app.py  
   ```  
   This starts a local server, allowing you to interact with the chatbot via a web interface.  

#### **Deploy on HuggingFace Spaces (Optional)**  
To make the chatbot publicly accessible, deploy it on HuggingFace Spaces. Upload the model to HuggingFace and follow their instructions to set up a space.  

---

## **File Structure**  

- **`app.py`**: Main Python script that initializes the Gradio app and handles chatbot interactions.  
- **`requirements.txt`**: List of dependencies required to run the application.  

---

## **Inference Pipeline**  

The fine-tuned model is loaded and used for inference through the following steps:  
1. **Tokenization**: User input is tokenized using HuggingFace's `AutoTokenizer`.  
2. **Model Generation**: The tokenized input is passed through the fine-tuned Llama-3.2-1B-Instruct model, leveraging LoRA to efficiently handle reduced memory requirements.  
3. **Streaming Output**: The generated response is decoded and sent back to the user in real time, ensuring smooth interactions.  

---

## **Fine-Tuning the Models**  

### **1. Model-Centric Approach**  
This approach focuses on optimizing the model's architecture, training algorithms, and hyperparameters. We fine-tuned the model's hyperparameters, such as learning rate, batch size, and weight decay. These adjustments impacted both convergence speed and training stability.  

Experiments included:  
- **Learning Rate Schedulers**: Linear, cosine, and constant.  
- **Weight Decay Values**: 0.0, 0.01, and 0.1.  
- **Batch Sizes**: 2, 4, 8, and 16.  

Further exploration of LoRA adapter hyperparameters, such as rank and scaling factor (alpha), was not implemented due to time constraints. These parameters could significantly affect performance metrics:  
- **Rank**: Higher rank increases trainable parameters but risks overfitting.  
- **Alpha**: Larger values amplify updates, improving convergence initially but risking instability.  
- **Dropout**: Helps prevent overfitting by randomly disabling connections during training.  

---

### **2. Data-Centric Approach**  
This approach focuses on improving data quality rather than modifying the model. High-quality data often yields greater benefits than model refinements. Techniques include:  
- **Strategic Subset Selection**: Choose a smaller, representative subset of data to focus on quality over quantity.  
- **Class Balancing**: Address underrepresented classes using oversampling methods like SMOTE, AdaSyn, or borderline SMOTE.  

---

