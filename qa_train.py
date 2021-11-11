# https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
import torch
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric

# dataset
model_checkpoint = "distilbert-base-uncased"
datasets = load_dataset("squad")
batch_size = 16                         # may need to be adjusted to avoid "out-of-memory" error

# ====== Preprocessing the training data ======
# tokenize the inputs and put into a format the model expects
# generate other inputs that model requires
# AutoTokenizer will ensure tokenizer corresponds to model architecture BERT and vocab is downloaded
# vocab will be cached so not downloaded next time it is ran
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# ensure tokenizer is a fast tokenizer - which is available for BERT
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

# question and answer text
question = "When did Einstein Die?"
with open('/home/katiew/Documents/einstein.txt') as f:
    answer_text = f.readlines()

# maximum length of a feature (question and context)
max_len = 384
# the authorized overlap between two part of the context when splitting it is needed
doc_stride = 128

# find long example in dataset
for i, example in enumerate(datasets["train"]):
    if len(tokenizer(example["question"], example["context"])['input_ids']) > max_len:
        break
example = datasets["train"][i]

# truncation MUST NOT be applied to question as we would loose information
tokenized_example = tokenizer(
    example["question"],
    example["context"],
    max_length=max_len,
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,     # gives each index of the inputs with (start, end) characters
    stride=doc_stride
)
# use mapping to find position of start and end tokens of our answer
first_token_id = tokenized_example["input_ids"][0][1]
offsets = tokenized_example["offset_mapping"][0][1]
# print(tokenizer.convert_ids_to_tokens([first_token_id][0], example["question"][offsets[0]:offsets[1]]))

# distinguish which parts of the offsets correspond to question and context
sequence_ids = tokenized_example.sequence_ids()
# 0's are the tokens for the question, 1's are the context, None are the special tokens
# print(sequence_ids)

# find first and last token of answer or determine answer dne
answers = example["answers"]
start_char = answers["answer_start"][0]
end_char = start_char + len(answers["text"][0])

# start token index of the current span in the text
token_start_index = 0
while sequence_ids[token_start_index] != 1:
    token_start_index += 1

# end token index of the current span in the text
token_end_index = len(tokenized_example["input_ids"][0]) - 1
while sequence_ids[token_end_index] != 1:
    token_end_index -= 1

# detect if the answer is out of the span - if it is, label with the CLS index
offsets = tokenized_example["offset_mapping"][0]
if (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
    # move the token_start_index and token_end_index to teh two ends of the answer
    # NOTE: could go after last offset if the answer is the last word (edge case)
    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1
    start_position = token_start_index - 1
    while offsets[token_end_index][1]>= end_char:
        token_end_index -= 1
    end_position = token_end_index + 1
    # print(start_position, end_position)
else:
    print("Answer not in this feature.")

# to account for the potential where the model expects padding on the left, switch question and context
pad_on_right = tokenizer.padding_side = "right"

# ====== Apply to training set ======
# in case of impossible answers, set CLS index for both start and end position
def prepare_train_features(examples):
    '''
        This function works with one or several examples
        In the case of several, will return a list of lists for each key
    '''
    # some questions have whitespace on left that can result in truncation of the context fail
    # so, remove whitespace on lhs
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # tokenize examples with truncation and padding - keep overflows using a stride
    # results in one example possible giving several features when a context is long
    # each of the features have context that overlaps a bit with previous feature
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # since one example might give several features if it has a long context,
    # we need a map from a feature to its corresponding example - this key does that
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset mappings will give us a map from token to character position in original context
    # will help compute start_positions and end_positions
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # label the examples
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # label impossible answers with the index of the CLS token
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example to know the context and question
        sequence_ids = tokenized_examples.sequence_ids(i)

        # one example can give several spans, this is the index of the example containing this span of text
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # if not answers are given, set the CLS index as answer
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # otherwise answers are found
            # set start/end character index of the answer in the text
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # start token index of the current span in the text
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # end token index of current span in text
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # detect if the answer is out of the span - if so, label this feature with CLS index
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # otherwise move token_start_index and token_end_index to the two ends of answer
                # NOTE: could go after the last offset if answer is the last word (in which we would have an edge case)
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# call prepare_train_features function for all sentences in our dataset
# features = prepare_train_features((datasets['train'][:5])) # single sentence
tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

# ====== Fine-tuning the model ======

