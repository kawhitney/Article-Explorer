# adapted from https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/
import torch
from transformers import BertTokenizer, BertForQuestionAnswering


class QuestionAnswering:
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    def predict_answer(self, question, context):
        # apply tokenizer to input text at text-pair (concatenated)
        input_ids = self.tokenizer.encode(question, context)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # search input_id for the first instance of the '[SEP]' token (until the question)
        sep_index = input_ids.index(self.tokenizer.sep_token_id)

        # number of segment A tokens includes the [SEP] token itself, remainder is segment B
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a

        # construct the list of 0s and 1s - 0s for question, 1s for context, none for special tokens
        segment_ids = [0] * num_seg_a + [1] * num_seg_b

        # feed example into model
        start_scores, end_scores = self.model(torch.tensor([input_ids]),  # The tokens representing input text
                                              # The segment IDs to differentiate question from answer_text
                                              token_type_ids=torch.tensor([segment_ids]),
                                              return_dict=False)

        # highlight answer by looking at most probable start and end words
        # find tokens with highest 'start' and 'end' scores
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        # combine tokens in the answer
        answer = ' '.join(tokens[answer_start:answer_end + 1])

        # reconstruct any words that were broken down into subwords
        answer = tokens[answer_start]
        # select remaining answer tokens and join with white space
        for i in range(answer_start + 1, answer_end + 1):
            # if a subword, recombine with previous token
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            else:
                answer += ' ' + tokens[i]

        # turn start_scores to list - to access prediction value
        start_scores = torch.flatten(start_scores).tolist()
        return [answer, start_scores[answer_start]]
