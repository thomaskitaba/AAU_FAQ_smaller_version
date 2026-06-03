# AAU_FAQs
# AAU_FAQ_smaller_version

<!-- Course Project 20%
Fantahun Bogale
•
May 30
20 points
Due Jun 5, 11:59 PM
In this project you are required to solve a reasonable NLP problem. 
You will train---evaluate--deploy your model at a prototype level with a public end-point. 
It is an Individual Project. 
Topics should not be repeated.
The class representative will compile the topics associated with the student and send me. 

Suggested Problems include:
Machine Translation (a local language to English or a local language to another local language)
Hate Speech Detection (in a local language)
Text summarization involving a local language.
Threatening speech detection from phone or any other conversation.
Conversational agent using a local language.
Fake News Detection in Amharic or other Local Languages 
Question-Answering System in Local Languages 
Plagiarism Detection System 
Speech-to-Text Transcription (STT)
Building a Sarcasm Detector  
Grammar Correction System  
Emotion Detection in Text  
Text Classification for News Articles (multiple classes)
You can add more (needs approval of the instructor) -->


### For the Final Code

Final code

1.app.py

This file will serve as the final application interface using Streamlit. It should be clean, well-commented, and easy to navigate. Ensure all functionalities are working as expected and the code follows the best practices we discussed. Include:

- All Streamlit code and imports.
- The final model pipeline (e.g., using `pipeline` for embedding + BiLSTM). Make sure it loads the model and tokenizer correctly.
- A clear user interface with input fields for questions and buttons for actions like search.
- Error handling for cases where the model might not load or an error occurs during inference.

2. test_model.py

- Verify the final optimized code with the cleaned dataset (FAQs with removed HTML/Junk and correct spellings).
- Include predictions and ground truth for at least 5 examples to demonstrate functionality.
- Compare results with the BiLSTM-only approach and show performance improvements.

3. Download_model.py

- You can delete this file as the model will be committed directly to the repository for better performance.


# Final CheckList

| Status | Section | Checkpoint | Notes |
| :---: | :--- | :--- | :--- |
| ☐ | Data | Dataset Cleaning | All HTML/Junk removed? Spellings corrected? |
| ☐ | Model Pipeline | Embedding + BiLSTM | Model loading correctly? Embedding + BiLSTM working? |
| ☐ | Performance | Comparison | Results with BiLSTM-only? Improvements shown? |
| ☐ | App Interface | Streamlit App | Clean UI? Input fields? Buttons? Error handling? |
| ☐ | Testing | Test Script | 5+ examples with predictions and ground truth? |
| ☐ | Documentation | Final Files | app.py, test_model.py cleaned? Download_model.py deleted? |
