# SpamClassification

This project is about classifying emails/sms texts as *spam* or *ham* using deep learning. Given that the data for this task is text data we will be utilizing the [huggingface transformers](https://huggingface.co/transformers/) library. This libraray provides pretrained tokenizer modules that eliminates the need to develop our own text preprocessing, and instead focus on the ML Ops apsects of the implementation. 

Data is collected from the Kaggle [Spam Text Message Classification] (https://www.kaggle.com/team-ai/spam-text-message-classification) dataset. This data is a collection of personal text messages and include many informal words. 

We will use an LSTM network as our classifier, as this type of model can be very good at handling sequential data because of it's recurrent structure. 

Group members:
- Simon Jacobsen, s152655
- Jakob Vexø, s152830
- Morten Thomsen, s164501
- Gustav Hartz, s174315


to get data:

run ```src/data/get_data.py```