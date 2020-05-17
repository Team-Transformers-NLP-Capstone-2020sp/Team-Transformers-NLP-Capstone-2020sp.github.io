# Team Transformers

Brock Grassy, Dan Le, Kaushal Mangipudi

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
  
<laughter> i don't know.
  
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
