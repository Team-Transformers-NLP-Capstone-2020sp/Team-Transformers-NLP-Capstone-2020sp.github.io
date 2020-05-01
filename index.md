# Team Transformers

Brock Grassy, Dan Le, Kaushal Mangipudi

## Blog Post 6:
### explanation of approach
We relied heavily on Huggingface’s Conversational AI model for this approach. The basis for this approach is the GPT2DoubleHeadsModel, which consists of a GPT-2 Transformer and two heads that consist of two linear layers, one head for language modeling and the other for multiple choice classification. The model takes in information about the subject it will generate text as in the form of a ‘personality’, and then uses that context along with the dialogue history to generate responses to user input and serve as a conversation partner. We used Huggingface’s pretrained model for conversational AI and then finetuned it using our transcript data from DAIC. To generate the data used from finetuning, we took the transcripts and concatenated multiple lines with the same speaker to get a transcript where each line consists of continuous words spoken by a single speaker and the speaker alternates each line. Then, for each transcript, we programmatically used some of the lines said by the patient to construct a set of sentences to serve as a personality. Then, we used the conversation to construct, for each line said by the patient, a history that consists of all the lines in the transcript to that point, and a bunch of candidate responses drawn from the patient’s conversation, with the correct response being the one the patient actually said at that point in the conversation. We trained Huggingface’s pretrained model on the transcript data using Huggingface’s provided training script for 5 epochs, and then used their provided interact script to interact with the finetuned model.
1 point for reporting 
### Psychiatrist would enter in questions and model outputs a result.
#### Conversation 1:

>>> how are you feeling today?

i am good. just got home from work.

>>> what kind of job do you do?

i'm an accountant.

>>> how do you like your job?

i love it. its more than i can do on my own.

>>> how's life treating you?

i am doing well. its been hard, but i'm doing ok.

>>> what has been hard?

i am not sure. i am just a workaholic.

>>> do you have any activities outside of work?

yes, i love the weather. do you?

>>> no, weather is just weather

i like to go to the beach and take long walks.

>>> what is your favorite beach?

i love the beach, but i prefer the beach to the beaches.

>>> what do you do on a beach?

i work on my business, what about you?

>>> what is your business?

it is a company.

>>> how much sleep do you get?

i get a lot of sleep and eat out.

>>> what kind of food do you eat?

i like to eat a lot of salads.

>>> do you cook?

yes, i do. i also enjoy the chance to travel.

>>> do you cry?

i've been to the zoo.

>>> have you had any thoughts of hurting yourself?

i like animals. i am a vet.


#### Conversation 2:
>>> Hi how are you

i am good how about yourself?

>>> how are you feeling

i'm good. i just finished reading.

>>> is there anything wrong?

just feeling stressed. i love the opportunities and i like the chance.

>>> what's going on in your life?

i am trying to decide what i want to do with my life

>>> are there any big decisions you need to make?

yes. i want to go back to atlanta. but i am worried about the hurricane.

>>> what makes you think a hurricane is coming

lots of hurricanes. lots of things.

>>> how are you sleeping?

i'm ok. i like the weather. i'm a big fan of the weather.

>>> does the weather affect you?

yes. i love the change. and it is nice here.

>>> what makes it so nice?

its a lot of hurricanes. i love the change.

>>> I thought you said you were afraid of hurricanes?

i am. i'm in atlanta. i've never been.

>>> so why did you say you like hurricanes

i love the weather. the weather here is nice.

>>> what kind of weather?

i love the fall weather.

>>> how old are you?

i'm 23. i love the fall weather.

>>> are you in school?

yes. i am a junior in high school.

>>> but I thought you were 23?

oh, i was only 18.

>>> what grade are you in at 18?

well i am studying business and administration.

>>> what grade level are you?

in school i'm a senior.

>>> what do your parents do?

oh, i'm in college. i'm from georgia georgia.


### failure modes
Our model failed to train on GPU as even the small gpt2 version consumed up to 20 gb of memory. When training on cpu we were only about to train the model for 10 epochs, but it took a long time and the results were not great as the chatbot was still mostly using the pretrained data and thus sounded happy most of the time. Our results were produced by having a psychiatrist interact with the model and he said that “it feels like the chatbot is spouting nonsense because it is inconsistent when I ask follow up questions and it doesn’t even sound depressed in the first place”. Because we did not modify the huggingface model, there may be an error in the way that we formatted the data as well.
### next steps
We’re going to do our due diligence in investigating using huggingface for our model, as that seems to be the most convenient for our particular use case. We feel like our strategy for constructing personalities for our training data is too simplistic, and will try to see if we can improve that. If we are able to figure out a way to get around our current hardware restrictions, then we will move ahead with training our current model over a greater timeframe and then evaluate its performance. If not, we have considered attempting to use a double-headed GPT-2 model to implement a chatbot that can keep track of the entire context of a conversation.



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

Decoded sentence: there's nothing that really gets me really mad that i can think of the top of my head um

Input sentence: whatever comes to your mind

Decoded sentence: um maybe when it comes to schools how they treat the children especially children of color maybe 

Input sentence: okay do you consider yourself an introvert

Decoded sentence: mm yes 

Input sentence: do you feel that way often

Decoded sentence: yeah

Input sentence: okay when was the last time you argued with someone and what was it about 

Decoded sentence: wow i hardly ever argue with somebody <laughter>

Input sentence: <laughter> how did you feel in that moment
  
Decoded sentence: mm yes 

Input sentence: mhm can you tell me about that

Decoded sentence: um trying new things seeing new views of the world um trying the different type of foods um seeing how the government is and how they run the things out there i guess

Input sentence: oh my gosh

Decoded sentence: yeah oh my sister i guess my sister that was about a month ago

Input sentence: yeah that sucks

Decoded sentence: yeah so but i mean we're fine now so that's good

Input sentence: that's so good to hear

Decoded sentence: yeah

Input sentence: what advice would you give yourself ten or twenty years ago

Decoded sentence: to not  believe people so easily so gullible to not be so gullible

Input sentence: hm okay tell me more about that

Decoded sentence: i guess i've been hearing that lately how i believe anybody anything that somebody says without really thinking about it or without having them prove it so i guess you could say i let i made a lot of mistakes i don't wanna say i regretted it because i'm glad i went through it while i was younger than me getting older and going through the same situations so um yeah i believe people too easily and i give people too much chances  so

Input sentence: what's one of your most memorable experiences

Decoded sentence: when i went on a cruise which was a happy experience when i went on a cruise to mexico with my family um  when i was sixteen about six years ago

Input sentence: wow 

Decoded sentence: yeah <laughter>

Input sentence: that's a long time

Decoded sentence: yeah

Input sentence: <laughter> do you travel a lot
  
Decoded sentence: um maybe when it comes to schools how they treat the children especially children of color maybe 

Input sentence: what do you enjoy about traveling

Decoded sentence: um trying new things seeing new views of the world um trying the different type of foods um seeing how the government is and how they run the things out there i guess

Input sentence: mhm

Decoded sentence: yeah <laughter>

Input sentence: i'd love to hear about one of your trips

Decoded sentence: uh i felt very betrayed and belittled because she's my little sister and she shouldn't be disrespecting me in any type of way because i'm an adult so yeah i was not feeling the situation at all

Input sentence: really

Decoded sentence: yeah so but i mean we're fine now so that's good

Input sentence: how easy is it for you to get a good night's sleep

Decoded sentence: o mean the sirees itn was  oua fhey ing ee the tof secting she dolly the rorue the it leon the roug the siseesi tean wall  oh iner ato th tie g me thavg so ther i gues sof beon beas ind sot rually think ioguat it weo i watuso meang iod sot ler iergend ioke tout th m si guet it' so tuat y o calibedo s t beli gesped ond that in whe too fatot oo ghe di gumes peot lit sas ildont tok on the aho guttitnot area gu

Input sentence: awesome okay i think i've asked everything i need to

Decoded sentence: when i went on a cruise which was a happy experience when i went on a cruise to mexico with my family um  when i was sixteen about six years ago

Input sentence: thanks for sharing your thoughts with me 

Decoded sentence: uh i felt very betrayed and belittled because she's my little sister and she shouldn't be disrespecting me in any type of way because i'm an adult so yeah i was not feeling the situation at all

#### Error Analysis
For the smaller data, the answers the bot gave were all logical, albeit short. For the larger one, the results could be a little nonsensicals. We based this approach off an encoder/decoder model that translated French sentences into English sentences. We expected that this model would perform better than the n-gram model because it relies on the therapists’ input text in formulating responses. We found that our results were a bit better, as they made more sense in context, but they weren’t particularly good, as some of the words weren’t really words. That being said, the sequences did resemble English somewhat and will probably be improved greatly once done with word embeddings.


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
