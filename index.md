# Team Transformers

Brock Grassy, Dan Le, Kaushal Mangipudi

## Blog Post 10:
Final Presentation video: https://youtu.be/VXhQ-d2ti7g

Final Paper: 
Team-Transformers-NLP-Capstone-2020sp.github.io/Team_Transformers__Final_Report_CSE_481_N.pdf 

## Blog Post 9:

### advancing solution attempt #2

We tried to improve our model by training for additional epochs and training with more candidates considered per sentence. We tried to increase the number of candidates considered from 2 to 4, but we got CUDA memory errors because the GPUs available to us weren’t large enough to fit more than 2 candidates per sentence said by patients. We also attempted to train the model for 20 epochs on our training data, but we ran into a CUDA memory error on the 14th epoch. We attempted to debug the issue, but were unable to determine what was causing this to occur.

We also decided to try a different approach when defining the sentiment inputs for the conversation. As discussed in our last blog post, our previous strategy involved calculating the sentiment score for each patient response throughout the conversation and discretizing this input into “positive”, “neutral”, and “negative” categories. After passing this additional information into our model and allowing the user to select what sentiment they want the agent’s response to align with, we found that this “knob” essentially worked to directly tweak the sentiment of the response in the corresponding positive or negative direction. For instance, if the user specifies they want to turn the knob to “negative”, responses would frequently be rife with words such as “no” that explicitly encode negative sentiment. This change didn’t prove to be entirely effective- as we described in the last blog post,occasionally answers that would be considered to have negative sentiment by the isolated NLP definition would not actually correspond to negative responses within the context of the entire conversation.

In an attempt to change this behavior, we modified the sentiment inputs to reflect the sentiment of the conversation as a whole. To accomplish this, for all conversations in our input data we found the proportion of messages with negative sentiment as defined by the Vader sentiment model in nltk. We then split these conversations into positive, negative, and neutral categories based on these proportions, and passed these inputs into our model. The goal of this strategy was to reflect how conversations typically remain consistent in mood throughout and prevent the volatile changes in sentiment that came by tweaking the knob in the previous format. As our sentiment inputs now reflect the mood of the entire conversation, we expected the bot to have more variance in the sentiment of individual messages while preserving an overall negative sentiment.

### results:

One of the hyperparameters passed into the model is the “conversation history” hyperparameter, which controls the number of previous messages in the conversation that the model has access to. Our previous experiments used 2 as the default number for this hyperparameter.  We ran with a conversation history hyperparameter of 4 instead of 2, as it was necessary to do so in order to include the new sentiment information that we were encoding. We proceeded to get psychiatrist input on the model with the new sentiment knobs trained on these hyperparameters:

#### Conversation 1: (Conversation sentiment is negative)
> are you sad?

yes.

> what is making you feel sad?

um my depression.

> how long have you been depressed?

um i've been depressed for about six months now.

> why are you depressed?

um just feel bad for myself.

> why?

because i don't have health insurance.

> do you have a job

uh just so i can afford it.

> do you have a job

um i do.

> what kind of work do you do?

cooking.

> what is your job?

i cook.

> do you cook at a restaurant?

yes i do.

> do you enjoy your job

i don't.

> what don't you like about the job

um the fact that i'm unemployed so i'm not able to afford to be a cook.

> what job would make you happy?

i don't know i would love to be a chef.


#### Conversation 2: (Conversation sentiment is negative)

> are you sad? 

yes. 

> why are you sad? 

i'm upset about not getting married.

 > why do you feel the need to get married? 
 
i'm upset about not having children.


#### Expert Feedback:
“This is close to a preliminary psychiatry session. It's not bad, especially if it is for casual call line therapy.”


### additional error analysis:

One thing we noticed is that the model performed a lot better when we tried to structure the questions we asked it to mimic the question structure of the questions asked by the therapist in the transcripts we trained the model on, like in this conversation:

> hi, how are you?

pretty good.

> where are you from?

mhm.

> are you sad?

yes.

> what do you like to do?

um i like to walk.

> where do you work?

mhm.

> where do you work?

uh in the office.

> what did you study?

business.

> why are you sad?

i don't know.

> would you go to therapy?

it would be too expensive.

> what is the situation?

uh just depressed.

In this transcript, the chatbot was able to accurately answer the questions we asked it without us having to repeat the questions. This suggests that the model struggles to generalize beyond the training data, which makes sense given our difficulty in acquiring more transcripts. According to our expert, licensed therapists and psychiatrists will ask open ended questions, and if something like depression comes up they will follow up on the specific situation. It is difficult to gain information about “why” the chatbot is sad or “how” it lost its job. Because the therapist in our training data isn’t specific when asking questions, responses are more conversational and don’t allow for a lot of depth. Overall, the chatbot performed a little bit better than last week as it now can at the very least make small talk with a psychiatrist.


### sketch of next action plan:
We will start focusing on preparing our chatbot to be released to other psychologists as a part of our evaluation. To do this, we’ll first need to set up a website with a front end that psychiatrists can use to interact with the chatbot. As it stands, our model doesn’t generalize beyond the questions asked in the training data that well. We’re looking into ways to direct therapists to ask questions that more closely resemble the ones asked in the transcripts we trained on, to improve the quality of the responses the chatbot gives them. We will need to be very careful in how we go about doing this. We need to give therapists sufficient freedom to ask the questions they want to ask, as overly restricting the scope of questions therapists can ask limits the power of the chatbot as a training tool for therapists. Then we will design a survey for psychiatrists to fill out once they’ve interacted with the model to collect metrics to evaluate our model with and identify further places for exploration. This requires careful thinking about what type of metrics we will use (such as a point score for how depressed the chatbot sounds). In addition, we will spend some more time attempting to debug the memory error caused by training the model for more than 10 epochs. If we are able to train the model for more epochs, we expect the model’s performance to improve. 


## Blog Post 8:
### explanation of approach

One of the issues that we were running into with our previous model is that it wasn’t accurately conveying the mood that we were anticipating it to- one would expect a chatbot trained on counseling conversations to display negative sentiments, but ours was neutral or positive on many topics. Part of this can be attributed to our data, as our dataset contains conversations with people in a variety of different mental states. As a result of this imbalance, only a small portion of our conversations were giving us the results we expected. We spent some time seeking out feedback about ways to handle this issue. The idea of creating an additional “knob” to directly control the sentiment was floated in both our group’s meeting with Kelvin and Noah’s comments after our presentation. We decided to look more into implementing this feature in our model’s pipeline. Kelvin sent us a paper from Google Research (https://arxiv.org/pdf/2004.14546.pdf) that used a similar strategy when working on interpretable text-to-text models. The goal of their research was to train models that output an explanation alongside their prediction. To accomplish this goal, if they wanted to include an explanation with the prediction they prepended the phrase “explanation:” to their input data. We chose to follow this general paradigm of concatenating extra information to our input data for our specific use case.

The first step in our new process was to encode sentiment information in our input data. For this purpose, we used the pretrained Vader sentiment analysis tool provided in the nltk module. That module provides a function that takes a sentence and outputs a “compound” sentiment score that’s between -1 and 1, with negative scores corresponding to negative sentiment and positive scores corresponding to positive sentiment. For each conversation in our training data, we calculated the sentiment score for each line uttered by the participant. For our initial iteration of the model, we broke down the sentiment scores into three categories: between -1 and -0.05 is “<negative>”, between -0.05 and 0.05 is “<neutral>, and between 0.05 and 1 is “<positive>”. These strings were then appended to the end of each line said by the participant in the conversation, with an additional placeholder token in between. For instance, for our previous setup the conversation was in the following format where speaker1 is the therapist and speaker2 is the participant:
[speaker1_1, speaker2_1, speaker1_2, speaker2_2, ..., speaker1_n]
The participant’s next response is speaker2_n. Suppose that our Vader sentiment model evaluated at speaker2_n gives us a score less than -0.05. Our modified input data would thus look like:
[speaker1_1, speaker2_1, ..., speaker1_n, <sentiment>, <negative>]
After training our model on this modified data, our method of interacting with the model also changed to reflect the inclusion of sentiment. At each point in the conversation where the user is allowed to say something, we also allow the option for the user to input one of the previously described sentiment codes (such as <negative>). This allows the user to “turn the knob” and modify the mood of the agent at each point in the conversation; for instance, if the user wishes the agent to exhibit negative moods throughout they are able to do so.


In addition to working on adding sentiment knobs, we also improved the quality of the data we are training the model on . To provide another summary of what the training data looks like, it consists of a series of conversations between a therapist and patients seeking mental health treatment. Each conversation consists of three pieces: a collection of descriptive sentences that contain information about the patient  (e.g. ‘I am a writer’, ‘I regret dropping out of school’, etc.) that constitute a persona, the actual conversation between the therapist and the patient, and for each actual sentence said by the patient, a series of alternative ‘candidate’ sentences that the model must distinguish the actual sentence from. For example, in response to the question ‘how are you feeling,’ the patient might have actually said ‘I’m feeling down because I can’t find a job,’ and alternative candidate sentences could be sentences like ‘I’m feeling great’ or ‘I’m feeling excited about talking with you’. The task the model is trained on is to, given a persona and a question the therapist has asked, distinguish the actual sentence the patient said from the alternative candidate sentences. In the aforementioned example, the training task would be to, given the persona of the patient in the conversation and the question ‘how are you’, predict that the patient said that they’re feeling down instead of feeling great or excited.


We had already improved the quality of our personas by handwriting personas for the transcripts in prior weeks. In this last week, we also took steps to improve the quality of the alternative candidate sentences we generated for conversations. Our initial strategy for generating candidate sentences was to, for each transcript, create a pool of responses that consisted of all the sentences a participant said over the course of the conversation. Then, for each sentence said by the participant, we would randomly select 4 sentences from the pool to serve as candidates. This approach did not work well because while the candidate sentences were coherent, they were mostly unrelated to the question the participant was trying to answer. Even worse, all the candidate sentences originated from the same persona, which means that the model was unable to rely upon the personas to help determine the actual sentence said by the participant. The end result was that the model we trained did not make good use of the persona it was given while answering questions, so its answers did not reflect the persona and weren’t consistent. For example, the persona the model is using might say that the person is a programmer, but the model might answer a question about its career by saying that it is a writer.


Our solution was to generate candidate answers to questions by utilizing answers by other patients to the same question. We noticed that most of the questions asked by the therapist across all the conversations were asked to multiple patients. So, we generated a pool of candidate questions by iterating through all the transcripts, and for each question asked by a therapist, we would create a set of all the different answers given by various patients to the question. For example, for the question ‘do you have any regrets,’ the set of candidate answers might consist of ‘I regret dropping out of school,’ ‘I regret getting married so early,’ and ‘I don’t really regret anything’. Then, when generating the training data, for each question the therapist asked, we would randomly select 4 candidate answers from the pool of answers that correspond to that question. The vast majority of questions therapists asked had at least 4 different answers to them from the 200 or so transcripts we have, but for those that don’t, we randomly selected answers from across the pool to supplement the existing answers to get 4 candidates.


### results:

#### Conversation 1

>>> are you sad?

yes.

>>> how are you feeling today?

i feel good about my life.

>>> do you have a job?

i have no job.

>>> are you in school?

yes.

>>> what do you study?

business.

>>> do you like school?

business.

>>> what grades are you getting?

business.

>>> how is the weather?

weather.

>>> are you sick?

uh no.

>>> how is your family doing?

well they're doing great.

>>> are you married?

yes.

>>> do you have kids?

*bot broke


#### Psychiatrist analysis:

This bot was inconsistent with his moods but gave a string of answers for school and studying. His response to sickness was good. family question is good.

#### Conversation 2:

>>> are you sad?

sentiment:

>>> negative
  
no.

>>> have you been crying?

sentiment:

>>> positive
  
yes.

>>> do you want to hurt yourself?

sentiment:

>>> positive
  
yes.

>>> what kind of thoughts?

sentiment:

>>> neutral
  
sniffle sniffle sigh sniffle
  
>>> do you want help?

sentiment:

>>> positive
  
yes.

>>> do you want to see a doctor?

sentiment:

>>> positive
  
yes.

>>> will you take medications?

sentiment:

>>> positive
  
yes.

>>> how are you feeling?

sentiment:

>>> negative
  
i'm feeling really bad but i'm okay.

>>> why are you sad?

sentiment:

>>> neutral
  
(laughter) i don't know.
  
>>> are you anxious?

sentiment:

>>> positive
  
yes.

>>> why are you worried?

sentiment:

>>> neutral
  
i'm nervous.

>>> is something bothering you?

sentiment:

>>> positive
  
i'm anxious.

>>> what is bothering you?

sentiment:

>>> neutral
  
i'm not sure i'm feeling really bad.

>>> have you been able to sleep?

sentiment:

>>> negative
  
no.

>>> are you waking up at night?

sentiment:

>>> negative
  
no.

>>> do you have trouble falling asleep?

sentiment:

>>> neutral
  
no.

>>> are you rested during the day?

sentiment:

>>> negative
  
no.

>>> have you been able do go to work?

sentiment:

>>> negative
  
no.

>>> what do you do at home all day?

sentiment:

>>> neutral
  
i don't know.

>>> are you cleaning the house?

sentiment:

>>> negative
  
no.

  
#### Psychiatrist analysis:

This bot with the sentiment works better because you give it a direction in how the responses will go. 

#### result analysis:
Overall, we made significant progress from last week. The changes we made to how we generate candidate sentences for transcripts helped address a lot of the issues we had with the chatbot not staying consistent with the persona when answering questions. Including the sentiment information in our data also noticeably changed our results. If we asked the same question several times in a row while only changing the sentiment input, the bot’s responses varied roughly as expected. 

### failure modes

We are still encountering lots of issues with the chatbot. Even though we are starting with a pre trained gpt2 model, it is only helpful for generating text based on previous text. When it comes to responding to questions that are not in our training data, the chatbot will sometimes repeat itself or say gibberish. Most likely this is a matter of not having enough training data. In addition, the sentiment knob sometimes does not have the intended effect. We currently are using nltk and that assigns scores based on word scores so it may be causing responses that aren’t exactly negative in context. For instance <positive> will answer yes and <negative> will answer no; this causes problems if the negative answer to the question is "yes". For example, if we asked the model to answer the question "are you depressed" with negative sentiment, the model would answer "no". Consistency is also still an issue which can be improved with training with more epochs. We found that this could potentially result in overtraining in previous iterations of our model, but as we have increased the number of labeled personas and improved our candidate quality we have found this to be less of a problem.

### next steps
Now that we have explored different ways to improve the data, such as grouping answer candidates by question and improving personalities, we will be focusing exclusively on improving the sentiment knobs. When the psychiatrist controlled the conversation, it had a much more natural flow and sounded depressed. It feels more natural for the model to use the same sentiment across the entire conversation, so we will modify our sentiment knob to reflect this. In addition we want to expand from just the three sentiment options (negative, neutral, and positive) to more options to see if we can have greater control over how depressed the chatbot sounds.



## Blog Post 7:

### advancing solution attempt #1
For our initial solution, we were unable to train the Huggingface model on our GPU without running into memory errors- training it for 10 epochs on CPU took an incredibly long time. Since the submission of our last blog post, we were able to figure out how to get the model to run on the lab-provided GPUs. The initial results after running on a greater number of epochs were promising, as the agent’s responses tended to be more coherent as a whole. However, we found that it was still lacking overall consistency across responses. For instance, the agent would provide contradictory messaging when asked the same question at different points within the same conversation. We attributed this behavior to problems with our automatically generated “personas”. 
To provide another summary of what our model’s input looks like, there are basically three pieces of information required for a conversation: the messages sent during the conversation (utterances), the true response and false candidate responses for the agent at each point in the conversation (candidates), and a series of sentences describing the agent’s “persona”. Our initial strategy for extracting personas involved automatically finding sentences that start with “I” or “my” and using those directly. However, that gave subpar results, as rarely people would say sentences that describe their “personality” in the simple, distilled way we wanted them to. As such, we turned to manual labeling and creation of personas for each person. Sample sentences that are part of these personas include “I am from new york”, “I moved to l_a for business”, and so on. 
We were able to label 20 of the conversations fully and used those in our sample model. Initial results from this model proved to be promising, as the model did a good job encoding the individual personalities. Results are naturally still not perfect, but they are greatly improved over our last ones. One issue that consistently came up, for example, is that the agents don’t sound as “depressed” as we might expect them to do. A further exploration of this trend and what can be done to improve results on this front is included in the later sections of this post.
1 point for describing additional experiments run
We tried training on different pretrained models, such as huggingface’s finetuned gpt model, pretrained gpt2 small, and pretrained gpt. The finetuned gpt model caused some GPU errors, but later we were able to remove the validation metrics and run gpt2 without errors. In addition we tried processing the transcripts differently to include less candidates for responses to conversation.  It didn’t affect our results because the “true” response we mostly what the model was trained on. In addition, we tried assigning every conversation the same personality text such as “I am depressed”. It improved the coherence of the responses but was less consistent in its answers. We also adjusted the number of training epochs


### additional error analysis 
#### Conversation log of chatbot with psychiatrist:
##### Selected personality: i'm from colorado.i'm a paralegal.i love l_a for all the opportunities.i want to be an attorney to help people.i don't like it when people aren't empathetic.i am a single mother.i'm proud of my daughter because she just graduated from u_c_l_a.i'm pretty stressed.i was harassed during a workplace dispute and now have p_t_s_d.i love travelling.

>>> how is your day going?

uh it's good.

>>> what made it a good day?

uh i was just told that my daughter graduated from u_c_l_a

>>> what is she studying ?

um she's a paralegal.

>>> how are you sleeping?

uh paralegal.

>>> how many hours of sleep do you get?

oh two a day.

>>> are you eating well?

no.

>>> why not?

oh well i'm a paralegal.

>>> is your job stressful?

uh yeah i am.

>>> what makes it stressful?

uh when people don't give you the right kind of work environment.

>>> what is your work environment like?

uh i'm pretty stressed out and um very stressed out.

>>> is there anyone stressing you?

no.

>>> is the nature of the work stressful?

no.

>>> is it the quantity of work?

not the.

>>> what makes it stressful?

um.

>>> what makes the work stressful?

uh that it's easy to disregard and just disregard you're actually doing something that's important that you

>>> how is work stressful?

uh it's like you're in a situation where you're trying to argue with a wall of people

>>> are you arguing with a lot of people at work?

yes i am.
>>> why are the workers arguing with you?

because you're trying to get away from them.

>>> why do the workers want you away from you?

because you're trying to get away from them.

>>> have you been in trouble with work?

no.
>>> did you tell your boss what is happening at work?

oh no <laughter> no i'm sorry.
  
>>> Why wouldn't you tell your boss your work problem?

oh uh what do you mean.

>>> do you like your coworkers?

uh they're rude sometimes but they're not.

>>> what did they say to you that was rude?

uh well i didn't mean to disrespect them.

#### Psychiatrist feedback:
"what makes it stressful? uh when people don't give you the right kind of work environment."
Were very good lines. Everything else was bad, and most of the personalities (conversations not listed here) were not depressed at all. Because therapy relies on follow up questions the history of the conversation needs to have more of an impact.


We ran the nltk sentiment analysis on the chatbot’s responses from the conversation and it output a compound score of 0.1328 which is somewhat positive leaning on the neutral side. Negative compound scores indicate negative sentiment while positive scores indicate positive sentiment. Overall our chatbot has improved its capabilities but there are many more improvements that need to be made. One thing that caused the chatbot to be inconsistent was that our handwritten prompts for the chatbot were too complex. It used the information in the wrong context, and completely ignored certain sentences. It is already hard for current chatbots to maintain conversation history so our problems with it may be a result of that, or made worse with the sparse amount of data that we are working with.

### sketch of next action plan
We haven’t finished annotating all the 175 transcripts yet, so we’ll try to get that finished ASAP. We think some of the issues we have might be caused by a lack of data, so finishing this and increasing the pool of annotated data should hopefully help with that.
Another thing we are considering is improving our methodology for providing candidate options for the bot to say. Our current approach is to provide all the things said by a depressed person in a conversation as candidates for each sentence, which doesn’t help train the model to give answers to questions that are specific to their persona. A lot of the questions asked by the bot are similar, so one idea is to create ‘pools’ of answers to specific questions to use as candidates. For example, one question commonly asked in the transcripts is ‘what caused you to seek help?’ One person who suffered a specific trauma in the past migchaht point to that particular incident, while someone who is just upset with their life in general might have a different response. Including both of their responses would help the model’s responses to questions like these more sensitive to what’s in the personality, which should help in making the model more self-consistent.
Another, alternative path we could take that we’re looking into right now is to create knobs that could tune how the model sounds, e.g. how cooperative the model is, how depressed the model should act, etc. This is an interesting idea that could help psychiatrists get practice in dealing with specific types of patients and get extra training with types of patients that they might find more difficult to deal with. To this, we’d need to further annotate our data on axises like cooperativeness/hostility and how depressed the patients are in the transcripts, but we don’t necessarily have to do this by hand - we could use existing tools like sentiment analysis to do this automatically. If we decide to pursue this line of inquiry further, we’ll first try automatically labeling the data using sentiment analysis and see if the annotations are of sufficiently high quality to use in training. If they are, we’ll try to use it training. We currently aren’t super optimistic about the feasibility of this approach, however, as we feel like the relatively small amount of data we have at our disposal means that further subdividing the data based off of factors like severity of depression might make the small buckets of data harder to effectively train on.

A final idea we’re considering that could help address the data issue is to switch from GPT-2 to GPT. This might seem counterproductive, but in addition to the pretrained GPT-2 and GPT models, Huggingface has a publically available finetuned chatbot model based off of GPT that seems to perform fairly well. If, instead of starting with the pretrained GPT model and finetuning it with the transcript data, we start with the already finetuned GPT model from Huggingface, and then finetune further using the transcript data, we might be able to futher improve our results. We aren’t sure if this seems like a reasonable (or ethical - if we do this, we will definitely fully credit Huggingface, but we’re not sure if this would constitute some faux pas regardless) thing to do, and would welcome feedback on this potential next step.


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
