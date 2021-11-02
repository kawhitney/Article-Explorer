# https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/#part-2-example-code
import torch
from transformers import BertForQuestionAnswering as QA
from transformers import BertTokenizer

# fine tuned pretrained model
model = QA.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# question and answer text
question = "When did Einstein Die?"
answer_text = "On 17 April 1955, Einstein experienced internal bleeding caused by the rupture of an abdominal aortic aneurysm, " \
              "which had previously been reinforced surgically by Rudolph Nissen in 1948. He took the draft of a speech he was preparing " \
              "for a television appearance commemorating the state of Israel\'s seventh anniversary with him to the hospital, but he did not live to complete it. " \
              " Einstein refused surgery, saying, \"I want to go when I want. It is tasteless to prolong life artificially. I have done my share; " \
              "it is time to go. I will do it elegantly.\"He died in Penn Medicine Princeton Medical Center early the next morning at the age of 76, " \
              "having continued to work until near the end. During the autopsy, the pathologist Thomas Stoltz Harvey removed Einstein's brain for " \
              "preservation without the permission of his family, in the hope that the neuroscience of the future would be able to discover what made Einstein so intelligent. " \
              "Einstein\'s remains were cremated in Trenton, New Jersey,and his ashes were scattered at an undisclosed location."

# apply tokenizer to the input text as a text-pair (concatenated)
input_ids = tokenizer.encode(question,answer_text)

'''
# BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id...
for token, id in zip(tokens, input_ids):
    
    # If this is the [SEP] token, add some space around it to make it stand out.
    if id == tokenizer.sep_token_id:
        print('')
    
    # Print the token string and its ID in two columns.
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')
'''
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# search input-id for the first instance of the '[SEP]' token (until the question)
sep_index = input_ids.index(tokenizer.sep_token_id)

# number of segment A tokens includes the [SEP] token itself
num_seg_a = sep_index + 1

# remainder is segment b
num_seg_b = len(input_ids) - num_seg_a

# construct the list of 0s and 1s
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# feed example into model
start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                                 return_dict=False)

# highlight answer by looking at most probable start and end words
# find tokens with highest 'start' and 'end' scores
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# combine tokens in the answer
answer = ' '.join(tokens[answer_start:answer_end+1])

# reconstruct any words that were broken down into subwords
answer = tokens[answer_start]
#select remaining answer tokens and join with white space
for i in range(answer_start+1, answer_end+1):
    # if a subword, recombine with previous token
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    else:
        answer += ' ' + tokens[i]

# print answer
print('Answer: "' + answer + '"')