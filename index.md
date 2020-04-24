# Team Transformers

Brock Grassy, Dan Le, Kaushal Mangipudi

## Blog Post 5:

The two baselines we opted to use for our chatbot are an n-gram model and a LSTM model.:

### N-gram:
Our n-gram model is a simple model that generates text using a trigram model trained on what the patients said in the therapy session transcripts. We anticipated that a depression chatbot build using this model would perform relatively poorly because it would not be able to rely on what the person chatting with it says when formulating responses. Our prediction ended up matching our results. Here are some of the therapist statement / chatbot response pairs we ended up generating while testing the chatbot:

Therapist: what are some things that make you really mad
Chatbot: mhm

Therapist: okay do you consider yourself an introvert
Chatbot: i studied abroad in madrid spain when i was watching on the environment uh certain things like car issues sometimes

Therapist: hm okay tell me more about that
Chatbot: it's more prestige it just is really minor fortunately my i was having um more so i did the best decisions probably of recent would um be a particular time uh so but i made some bad experiences but for example he um derives a lot of congestion and um you know it's just dreams at least not for sure you know we had went to san francisco same custom and everything

Therapist: awesome do you travel a lot
Chatbot: yeah um jumping into a you know always circumstances in my life

#### Error Analysis
As can be seen, there are two main issues with the n-gram response. Firstly, there’s no way to determine the semantics present in the initial question/statement from the therapist, and the response has nothing to do with what the therapist said in the first place; in all cases the response is randomly generated through the same process based solely on initial n-gram frequencies. We also noticed in certain cases that the n-grams could result in arbitrarily long, incoherent concatenation of various responses. As such, the n-gram model doesn’t seem like the ideal model for our ending goal.

### LSTM:
Our second baseline approach is a more complex approach that relies on a LSTM model. The LSTM model uses an encoder decoder architecture that was trained on processed transcripts of therapy sessions between therapists and depressed patients. The model takes the input text that the therapist typed, encodes it using the encoder, and then uses the decoder to decode the encoded therapist text into client responses. We based this approach off an encoder/decoder model that translated French sentences into English sentences. We expected that this model would perform better than the n-gram model because it relies on the therapists’ input text in formulating responses. We found that our results were a bit better, as they made more sense in context, but they weren’t particularly good, as some of them were quite wacky. Because the encodings were done on a character to character basis, there is a lot of nonsense mixed in. Thus in our next iteration we plan on making it a word to word translation model by encoding the words through an embedding layer. That way we will get full words. Here are some examples:

#### Small dataset results:
Input sentence: awesome do you travel a lot
Decoded sentence: no

Input sentence: i'm sorry
Decoded sentence: [laughter] it's alright

Input sentence: how do you know them
Decoded sentence: friendship

Input sentence: cool how would your best friend describe you
Decoded sentence: chocolate tall

#### Large dataset results:
Input sentence: what are some things that make you really mad
Decoded sentence: there s the i gher ing whin s ae se in an s o  u                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

Input sentence: mhm
Decoded sentence: yeah
                                                                                                                                                                                                                                              
Input sentence: why
Decoded sentence: yeah

Input sentence: do you feel that way often
Decoded sentence: yeah                                                                                                            

Input sentence: oh my gosh
Decoded sentence: yeah

Input sentence: that's so good to hear
Decoded sentence: yeah  o m t i e r                                                                                                                                                                                                                                                                                                                                                                                                                                              

Input sentence: hm okay tell me more about that
Decoded sentence: um myye hhe witting s ie ghe wholllthet rereald m m mind an   w i                                                                                                                                                                                                                                                                                                                                                                                               

Input sentence: hm
Decoded sentence: yeah                                                                                                                                                                               

Input sentence: wow
Decoded sentence: yeah <laughter>
  
Input sentence: that's a long time
Decoded sentence: yeah s m be i ie w a                                                                                                                                                                                                                                                                                                                               

Input sentence: mhm
Decoded sentence: yeah

Input sentence: i'd love to hear about one of your trips
Decoded sentence: mmey s i wet in a cases i whes i wathay aod reret in casesi go d alli ther i gue s i meas ald out me than  a  t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

Input sentence: mhm
Decoded sentence: yeah

#### Error Analysis
For the smaller data, the answers the bot gave were all logical, albeit short. For the larger one, the results were kind of nonsensical, as the responses. We based this approach off an encoder/decoder model that translated French sentences into English sentences. We expected that this model would perform better than the n-gram model because it relies on the therapists’ input text in formulating responses. We found that our results were a bit better, as they made more sense in context, but they weren’t particularly good, as some of the words weren’t really words. That being said, the sequences did resemble English somewhat and will probably be improved greatly once done with word embeddings.


## Blog Post 4:

### New Project Plan: 
* Create a chatbot that will help train counselors by emulating someone who is depressed. Stretch goal is to give the counselor an evaluation of how well they did and possibly how they can improve. Dataset is still the same.

### Approach and Experiment Explanation:
* For our strawman we will use twitter and or reddit data until DAIC becomes available.
* We take tweets that both were tagged with a depression related hashtag (#depression, #mentalhealth, #anxiety, etc.) a negative sentiment and use an trigram model to emulate those tweets (https://github.com/AshwanthRamji/Depression-Sentiment-Analysis-with-Twitter-Data).
* This will have similar output to our chatbot.
* We acknowledge that this is not the best dataset to use. We have submitted requests for access to other datasets in addition to DAIC and are optimistic that at least one one of them will become available to us.

### Example Results:
* rt @alyciatyre : my heart goes out to those of us whose anxiety has gone from crippling to an accelerating vomit / shit / deathlike ever since …
* depression , but i don't wish them upon anybody .
* t 😂 😂 😋 by alot of using their minds 😂 😂
* i have depression " just because i'm thinking of something more important to me feeling so tired and sick lately i haven't been able to go in @checkpointorg ' s kami dvorakova …
* post 1am depression twitter https://t.co/ibmpn4kn8l
* rt @kbelliard_ : nothing hurts more than depression , they're not thinking about life , you know what's really fun about bipolar disorder - anxiety is acting so badly rn lmao
* rt @playstationau : 24 hours to go to a therapist every month and always have panic attacks and anxiety https://t.co/gmhp3ldode
* my parents feel like there's something missing and that's why my depression ? https://t.co/tg3oxxndgl
* talkin bout cancer 😩 😩 😩 😩 😩 😩 😩 😩 😩
* i don't wanna talk until tomorrow 🎶

### Evaluation Framework:
* For now we will simply see if the chatbot sounds naturally depressed
* When we get full counseling data we can also evaluate the chatbot’s responses to the user based on sentiment analysis.
* We can also use an established sentiment dictionary to evaluate how negative the chatbot is

### Evaluation of results:
* Even with a simple n-gram model the tweets are pretty believable. There are a few nonsensical sections but overall most of them probably wouldn’t be distinguishable from regular tweets of this nature. The next step is to use reddit data which will be more grammatical so the n gram may not work as well.


## Blog Post 3:

### Minimal Action Plan:
* Our final goal will be to produce a chatbot that will have a brief conversation with a user and determine what issues they are facing. To reach that point, we will stick to the following plan:
* Reach out to Tim Althoff to touch base about his research and receive input about our project (as well as potential ethical concerns/questions)
* Using the data provided the crisis line, train a LSTM model that can use helpline conversations to predict what issues the caller is facing
* Do an analysis of our results to determine what questions are the most significant in predicting what issues callers are facing
* Based off the analysis and existing medical questionnaires, train a model that uses the answers to previous questions to ask a question from a pool of hard-coded and predetermined questions that will best help determine what issue the caller is experiencing.

### Stretch Goals:
* Develop a chatbot for depression detection
* Move beyond diagnostics and attempt to apply therapeutic techniques
* Train a more sophisticated model that will help ask good leading questions that will help predict what issues the callers are facing (potentially using GPT-2)

### Motivation:
* Mental health is a major problem that needs to be addressed. We are making strides in improving accessibility to treatment and quality of care, but there simply isn’t enough manpower available to address the needs of everyone who needs mental health care. An automated preliminary screening for people dealing with mental health crises that can identify what problems people are dealing with would greatly reduce the workload help lines have, and will allow them to quickly point people to issue-specific resources they might need.

### Related Work (literature survey):
* Analysis of counseling conversations (Althoff): https://arxiv.org/pdf/1605.04462.pdf
* Screening internet users for depression: https://www.sciencedirect.com/science/article/abs/pii/S0169260715000620
* Depression in social media: https://www.aaai.org/ocs/index.php/ICWSM/ICWSM13/paper/view/6124/6351
* Depression questionnaire: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371338/
* Indicators of depression: https://www.aclweb.org/anthology/W17-3101.pdf
* Detecting depression in interviews: https://groups.csail.mit.edu/sls/publications/2018/Alhanai_Interspeech-2018.pdf

### Project Objectives:
* Be able to detect symptoms of depressions based on a user’s speech
* Create a chatbot that will facilitate conversation and determine best course of action

### Proposed Methodologies:
* Train on Crisis Text Line data for issue differentiation using different embeddings and out of the box models.
* Make a model for determining what questions to ask based on the conversation.
* Create the chatbot that will dynamically ask questions and categorize crisis issues

### Available Resources
* Helpline transcripts: http://transcripts.cnn.com/TRANSCRIPTS/1503/31/csr.01.html
* Crisis line text messages: https://www.crisistextline.org/data-philosophy/ 
http://snap.stanford.edu/counseling/

### Evaluation Plan
For evaluating our first model that simply attempts to classify which issues a user is facing given a conversation, we can stick to standard metrics for classification quality (precision, recall, etc.)
In Tim Althoff’s paper “Large-scale Analysis of Counseling Conversations: An Application of Natural Language Processing to Mental Health”, he provides various methodologies that are useful in quantifying the efficacy of various counseling sessions and conversations. We could potentially leverage these techniques when moving to our goal to ask questions to the user, as it’s important to verify that our bot is not causing any harm.

## Blog Post 2:

### Citizenship NLU:

#### Pros: 
* Denise has already looked into the project so we would be able to receive support from  her. 
* Could potentially be useful to a large population of users, and clearly has real-life stakes.

#### Cons: 
* Target users might have a hard time typing responses and using technology in general
* Might not have time to implement speech to text

#### Codebase: 
* Denise starter code,  https://www.aclweb.org/anthology/R13-1006.pdf (Standford Named Entity Recognition)

### Sarcasm: abandoned.

### Refined Sentiment Analysis down to Psychotherapy Chatbot: 
* Have a conversation with a user that serves a preliminary screening for depression

#### Pros: 
* Mental illness is not well studied in NLP and there are lots of unexplored paths we could take in this project. 
* Mental health is a serious problem, and there are often systematic barriers for access to mental health care, so a project like this would have real stakes and could do a lot of good if it is successful.

#### Cons: 
* Lack of data (due to the sensitive nature of counseling conversations). 
* Costs of failure are very high, as the chatbots will be interacting with sensitive and vulnerable people - saying the wrong thing could significantly worsen users’ mental health.

#### Codebase: 
* https://www.isca-speech.org/archive/Interspeech_2018/abstracts/2522.html (LTSM model)

### Topics in Lecture: 
* GPT-2, knowledge extraction, Guest lectures from researchers across the field, Word Embeddings, Sentiment Analysis


## Blog Post 1:

Project Repo at https://github.com/Team-Transformers-NLP-Capstone-2020sp/Capstone-Project
### Project Ideas:

#### Citizenship NLU:
* Experiment with or replicate relevant models
* Look at starter code and existing applications
* Look at test and find user experiences for preparing for citizenship test
* Create or find data set with citizen test questions and answers
* Start with text input
* Give advice on how to answer better
* Stretch goal: Add support for a voice to text interface. Set up a 1-800 number that people can call and interact with, and add in language detection if there is time to allow us to direct callers to resources in their language if they don't know English.

#### Sarcasm:
* Replication study
* https://towardsdatascience.com/sarcasm-detection-with-nlp-cbff1723f69a
* https://github.com/anebz/papers#sarcasm-detection (linked in article, summary of literature on sarcasm detection)
* Examine existing models for sarcasm detection to get sense of state of the art performance
* Find dataset with labeled texts for input into model
* Determine which medium we want to examine and what context to provide (for instance if we’re looking at tweets, can we encode information about earlier replies in the chain, earlier user behaviors, and so on)
* Train model that ideally classifies texts on whether they contain sarcasm or not, compare results to state of the art/existing results
* Stretch goals include finding ways to extend existing models to encode relevant information that may be encapsulated in other models/with other information. This is vaguely nebulous at the moment but would hopefully become more clear the more we work on the project as we determine what is lacking. Perhaps some sort of regimented error analysis could be useful for this purpose.

#### Sentiment detection:
* Replication study of existing models for detecting emotion/mood of sentences
* https://www.sciencedirect.com/science/article/abs/pii/S0169260715000620
* https://www.researchgate.net/profile/Chetashri_Bhadane/publication/277564782_Sentiment_Analysis_Measuring_Opinions/links/55e5863508aebdc0f589e12e.pdf
* Experiment with ways to capture sentiment in embeddings. Focus on detecting emotional states such as depression.
* Find datasets for how to answer or respond based on mood. Stretch goal is depression chatbot
* Train model on responding to emotional states (stretch goal)


### Contact Info
led1@uw.edu
