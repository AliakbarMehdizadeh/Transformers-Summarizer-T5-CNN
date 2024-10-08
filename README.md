# Text Summarizer Using T5 Fine Tuned on CNN/DailyMail Dataset

This repository contains an implementation of a text summarization model based on the T5 (Text-To-Text Transfer Transformer) architecture. The primary goal of this project is to develop an efficient system capable of generating concise, human-readable summaries from lengthy news articles, making it easier to digest key information. This project specifically utilizes the CNN/DailyMail dataset, which is widely used for training models on the task of summarizing long-form text into brief highlights.

### Key Features:

1. Model Architecture: This project leverages the T5 model, a transformer-based architecture designed for various text generation tasks, including summarization. The model is trained to convert a long text input (news article) into a concise output (summary).
2. Dataset: The model is trained on the CNN/DailyMail dataset, which consists of news articles and their respective highlights, making it an ideal dataset for summarization tasks.
3. API Integration: An API is created for real-world use, allowing users to send documents or text inputs and receive summarized versions. This API is built using Flask (or FastAPI/Streamlit depending on your preference) for easy integration into applications.

   ![Sample Output](sample_usage.png)


### Usage

1. Clone the repository:
2. Create and activate a virtual environment
3. `pip install -r requirements.txt`
4. `python main.py` for training and saving the fine tuned model
6. Start the FastAPI server: `uvicorn app:app --reload`
7. Start the Streamlit app in a new terminal: `streamlit run streamlit_app.py`
8. Open your browser and go to `http://localhost:8501` to access the Streamlit app.

 
