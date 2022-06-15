#!/usr/bin/env python
# coding: utf-8

import numpy as np
import spacy
import pandas as pd
import regex as re
# def main():


# get_ipython().system('spacy download en_core_web_sm')

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv('data/ner_tagged.tsv', sep="\t")

data.rename(columns = {'O':'Tags','-DOCSTART-':'Tokens','-X-':'X'}, inplace = True)

for i, row in data.reset_index().iterrows():
    if type(row.Tokens) == str:
        if row.Tokens.lower().startswith("un") and "-N-" in row.Tags and row.Tokens != "undermounted":
            data.at[i,'Tags'] = row.Tags.replace("N-", "")
    if type(row.Tokens) == str:
        if row.Tokens.lower().endswith("less") or row.Tokens.lower() == "screw" or row.Tokens.lower().endswith("less.") and "-N-" in row.Tags:
            data.at[i,'Tags'] = row.Tags.replace("N-", "")

is_negative = []
for i, row in data.reset_index().iterrows():
    if type(row.Tokens) == str:
        if "-N-" in row.Tags:
            is_negative.append(True)
            data.at[i, "Tags"] = row.Tags.replace("N-", "")
        else:
            is_negative.append(False)
    else:
        is_negative.append(np.NaN)


data["is_negative"] = is_negative


def get_triples(df):
    """Takes a df of the form given by the dev data csv and an is_negative column and outputs a list of lists.
    Each list represents a sentence whose members are tuples of the form (word, iob, negation)"""
    
    current_sent = []
    
    all_sents = []
    
    for i, row in df.reset_index().iterrows():
        if type(row.Tags) == str:
            current_sent.append((row.Tokens, row.Tags, row.is_negative))
        else:
            all_sents.append(current_sent)
            current_sent = []
            
    return(all_sents)



def change_to_io(triples):
    """Given a list of sentences output by `get_triples`, replace B-tags wiith I-tags."""
    for sent in triples:
        for i, word in enumerate(sent):
            if word[1].startswith("B"):
                sent[i] = (word[0], "I" + word[1][1:], word[2])
                
    return triples


all_sents = change_to_io(get_triples(data))

neg_sents = []

for sent in all_sents:
    for word, iob, neg in sent:
        if neg:
            neg_sents.append(sent)
            break

def transform_sentence(sent, for_evaluation=False):

    """
    If for_evaluation=False: Given a sentence prepared with `get_triples`, outputs a list of lists;
    each list is a series of continuous words belonging to the same Outside portion of the sentence
    or to the same product/attribute/etc. Also outputs a list of indices, each index corresponding
    to a sublist that contains the tokens corresponding to an entity.
    
    >>> transform_sentence([("This", "O", False), ("is", "O", False), ("not", "O", "False"), ("a", "O", False), ("red", "I-COLOR", True), ("shirt", "I-PRODUCT", False)])
    [["This", "is", "a"], ["red"], ["shirt"]], [1, 2]

    If for_evaluation=True: Returns a list of positive entity names and a list of negative entity names.

    >>> transform_sentence([("This", "O", False), ("is", "O", False), ("not", "O", "False"), ("a", "O", False), ("red", "I-COLOR", True), ("shirt", "I-PRODUCT", False)],
    for_evaluation=True)
    ["shirt"], ["red"]
    """
    
    list_counter = 0

    final_list = []

    current_list = []

    entity_indices = []
    
    pos_indices = []
    
    neg_indices = []
    
    pos_ent_names = []
    
    neg_ent_names = []
    
    previous_iob = None
    
    for i in range(len(sent)):
        word = sent[i][0]
        iob = sent[i][1]
        if for_evaluation:
            neg = sent[i][2]
        
        if previous_iob == iob or i == 0:
            current_list.append(word)
            
        else:
            final_list.append(current_list)
            current_list = [word]
            if previous_iob.startswith("I"):
                entity_indices.append(list_counter)
                if for_evaluation:
                    if previous_neg == False:
                        pos_indices.append(list_counter)
                    else:
                        neg_indices.append(list_counter)
            list_counter += 1
            
        previous_iob = iob
        
        if for_evaluation:
            previous_neg = neg
            
    if final_list == []:
        final_list.append(current_list)
        if sent[-1][1].startswith("I"):
            entity_indices.append(list_counter)
            if for_evaluation:
                if sent[-1][2] == False:
                    pos_indices.append(list_counter)
                else:
                    neg_indices.append(list_counter)
    elif current_list != [] and current_list != final_list[-1]:
        final_list.append(current_list)
        if sent[-1][1].startswith("I"):
            entity_indices.append(list_counter)
            if for_evaluation:
                if sent[-1][2] == False:
                    pos_indices.append(list_counter)
                else:
                    neg_indices.append(list_counter)

    if for_evaluation:
        for ent_idx in pos_indices:
            for i, phrase in enumerate(final_list):
                if i == ent_idx:
                    pos_ent_names.append(" ".join(word for word in final_list[i]))

        for ent_idx in neg_indices:
            for i, phrase in enumerate(final_list):
                if i == ent_idx:
                    neg_ent_names.append(" ".join(word for word in final_list[i]))

        return pos_ent_names, neg_ent_names
    
    else:
        return final_list, entity_indices


def prepare_gold_data(sent_list, output_style="tags", word_of_interest=None):
    """
    Given a list of sentences prepared with `get_triples`:
    If `output_style="tags"`, outputs a list of lists, with each list representing a sentence as a sequence of
    True (negated) or False (not negated) at each sentence index.

    If `output_style="entities"`, outputs `pos_ents` and `neg_ents`, each of which contains a list of positive and negative
    entities for each sentence.

    If `word_of_interest` is supplied, filters the sentences to ones including that word (as a string).
    """
    if word_of_interest:
        active_list = []
        for sent in sent_list:
            for word, iob, tag in sent:
                if word.lower() == word_of_interest:
                    active_list.append(sent)
                    break
    
    else:
        active_list = sent_list
    
    if output_style == "tags":
        outputs = []
        for sent in active_list:
            out = []
            for word, iob, neg in sent:
                out.append(neg)
            outputs.append(out)
        return outputs
    
    elif output_style == "entities":
        pos_ents = []
        neg_ents = []
        
        for sent in active_list:
            sent_pos_ents, sent_neg_ents = transform_sentence(sent, for_evaluation=True)
            pos_ents.append(sent_pos_ents)
            neg_ents.append(sent_neg_ents)
            
        return pos_ents, neg_ents


def evaluate(preds, gold, input_style="tags", verbose=False):
    """Calculates the F1 score and other metrics for predictions.
    If `input_style="tags"`, preds and gold are a list of sentences represented as a sequence of True (negated)
    and False (non-negated).
    If `input_style="entities"`, preds and gold each contain two lists: the positive entities and the negative entities for each sentence.
    """
    
    total = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
    if input_style == "tags":
        for i, pred in enumerate(preds):
            printed = False
            for j, tag in enumerate(pred):
                total += 1
                if tag == True and gold[i][j] == True:
                    true_pos += 1
                elif tag == True and gold[i][j] == False:
                    false_pos += 1
                    if verbose and printed == False:
                        print()
                        print("Sentence number:", i)
                        print()
                        print("Predicted sequence:", pred)
                        print("Actual sequence:", gold[i])
                        printed = True
                elif tag == False and gold[i][j] == True:
                    false_neg += 1
                    if verbose and printed == False:
                        print()
                        print("Sentence number:", i)
                        print()
                        print("Predicted sequence:", pred)
                        print("Actual sequence:", gold[i])
                        printed = True
                elif tag == False and gold[i][j] == False:
                    true_neg += 1
                
    if input_style == "entities":
        pred_pos_ents = preds[0]
        pred_neg_ents = preds[1]
        gold_pos_ents = gold[0]
        gold_neg_ents = gold[1]
        
        total = len([pred for preds in pred_pos_ents for pred in preds]) + len([pred for preds in pred_neg_ents for pred in preds])
        
        for i in range(len(pred_pos_ents)):
            false_positives = []
            false_negatives = []
            for ent in pred_pos_ents[i]:
                if ent in gold_pos_ents[i]:
                    true_neg += 1  # Correctly predicting a positive (i.e. non-negative) entity is a true negative since "negated" is treated as the positive class.
                    gold_pos_ents[i].remove(ent)
                else:
                    false_neg += 1
                    if verbose:
                        false_negatives.append(ent)
            
            for ent in pred_neg_ents[i]:
                if ent in gold_neg_ents[i]:
                    true_pos += 1
                    gold_neg_ents[i].remove(ent)
                else:
                    false_pos += 1
                    if verbose:
                        false_positives.append(ent)
                            
            if verbose:
                if false_positives or false_negatives:
                    print("Sentence number:", i)
                if false_positives:
                    print("False positives:", false_positives)
                if false_negatives:
                    print("Missed:", false_negatives)
    
    print("true pos", true_pos)
    correct = true_pos + true_neg
    accuracy = correct / total
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("Evaluation config:", input_style)
    print("Accuracy:", "{:.2%}".format(accuracy))
    print("Precision:", '{:.2%}'.format(precision))
    print("Recall:", '{:.2%}'.format(recall))
    print("F1:", '{:.2%}'.format(f1))




def get_spacy_tokens(transformed_sent, entity_indices):
    """Maps entity indices to indices in a spacy object.
    `transformed_sent` and `entity_indices` are the two outputs of `transform_sentence`."""
    
    ent_2_spacy = {}
    string_sent = " ".join([item for sublist in transformed_sent for item in sublist])
    doc = nlp(string_sent)
    
    spacy_special = ["'", '"', ":", ";", ",", "?", "!", ".", "n't", "'m", " "]
    
    for ent_idx in entity_indices:
        substring = " ".join([item for item in transformed_sent[ent_idx]])
        for char in spacy_special:
            if substring.endswith(char):
                substring = substring[:-len(char)]
        
        if str(doc).count(substring) == 1:
            # end = str(doc).index(substring) + len(substring)
            end = str(doc).index(substring) + len(substring)
            if substring.endswith(" "):
                substring = substring[:-1]
            try:
                start = end - len(substring.split()[-1])
            except:
                print(transformed_sent)
                print(entity_indices)
                print(end)
                print(substring.split())
#             print(str(doc)[start:end])
            span = doc.char_span(start, end)
            n = 0
            while not span and n<4:
                span = doc.char_span(start, end+n)
                n += 1
            while not span and n<4:
                n = 0
                span = doc.char_span(start-n, end)
                n += 1
            if not span:
                print(transformed_sent)
                print(entity_indices)
                print(substring)
            else:
                span = span[0]
            index = span.i

        elif str(doc).count(substring) > 1:
            best_guess_distance = 10000
            original = len(" ".join([item for sublist in transformed_sent[:ent_idx] for item in sublist])) + 1
            for idx in [_.start() for _ in re.finditer(substring, string_sent)]:
                if abs(idx - original) < best_guess_distance:
                    best_guess_distance = abs(idx - original)
                    best_guess = idx
            end = best_guess + len(substring)
            start = end - len(substring.split()[-1])
            span = doc.char_span(start, end)[0]
            index = span.i
        
        ent_2_spacy[ent_idx] = index
        
    return ent_2_spacy

def predict_one_sentence(sent, negators, output_style="tags", word_of_interest=None, verbose=False):
    """Given a sentence provided by `get_triples` and a list of negator functions, make a prediction for the sentence.
    If `output_style="tags"`, outputs a sequence of True (negated) and False (non-negated).
    If `output_style="entities"`, outputs a list of positive entities and a list of negative entities.
    """
    
    if len(sent[0]) == 3:
        active_sent = []
        for word, iob, neg in sent:
            active_sent.append((word, iob))
    else:#new
        active_sent = sent        
    if word_of_interest == None:
        found = True
            
    else:
        found = False
        
    for word, iob in active_sent:
        if word.lower() == word_of_interest:
            found = True
            break
    
    if found:
        
        transformed_sent, entity_indices = transform_sentence(sent)

        spacy_mapping = get_spacy_tokens(transformed_sent, entity_indices)

        if output_style == "tags":
            out = [0] * len(sent)
            for negator in negators:
                neg_indices = negator(transformed_sent, spacy_mapping, output_style)
                if verbose:
                    for index in neg_indices:
                        print("Index", index, "negated by", negator.__name__)
                for idx in neg_indices:
                    out[idx] += 1

            for i, tag in enumerate(out):
                if tag == 0:
                    out[i] = False
                else:
                    out[i] = True

            return out

        if output_style == "entities":
            pos_ents = []
            neg_ents = []
            for negator in negators:
                # Note that the below gives indices of entities in transformed_sent, not the indices themselves
                negator_pos_ents, negator_neg_ents = negator(transformed_sent, spacy_mapping, output_style=output_style)
                for ent_idx in negator_pos_ents:
                    if ent_idx not in pos_ents and ent_idx not in neg_ents:
                        pos_ents.append(ent_idx)
                for ent_idx in negator_neg_ents:
                    if ent_idx in pos_ents:
                        pos_ents.remove(ent_idx)
                        neg_ents.append(ent_idx)
                    elif ent_idx not in neg_ents:
                        neg_ents.append(ent_idx)

            pos_ent_names = []
            neg_ent_names = []

            for ent_idx in pos_ents:
                for i, phrase in enumerate(transformed_sent):
                    if i == ent_idx:
                        pos_ent_names.append(" ".join(word for word in transformed_sent[i]))

            for ent_idx in neg_ents:
                for i, phrase in enumerate(transformed_sent):
                    if i == ent_idx:
                        neg_ent_names.append(" ".join(word for word in transformed_sent[i]))

            return pos_ent_names, neg_ent_names


def predict_sent_list(sent_list, negators, output_style="tags", word_of_interest=None):
    """Creates a list of predictions by calling `predict_one_sentence` on each sentence with the appropriate `output_style`.
    If a single-word string is supplied to `word_of_interest`, filters sentences in the list of sentences to ones which include that word."""
            
    if output_style == "tags":
        
        preds = []
        
        for sent in sent_list:
            sent_preds = predict_one_sentence(sent, negators, output_style, word_of_interest)
            if sent_preds is not None:
                preds.append(predict_one_sentence(sent, negators, output_style, word_of_interest))
            
        return preds
            
    if output_style == "entities":
        all_pos = []
        all_neg = []
        
        for sent in sent_list:
            preds = predict_one_sentence(sent, negators, output_style, word_of_interest)
            if preds is not None:
                pos_ents, neg_ents = preds
                all_pos.append(pos_ents)
                all_neg.append(neg_ents)
            
        return all_pos, all_neg


def get_negator_output(transformed_sent, entity_indices, negated_indices, output_style):
    """A helper function that gets the input into the right format for each of the negator functions (see below)."""
    
    if output_style == "tags":
        preds = []
        k = 0
        for i, sublist in enumerate(transformed_sent):
            if i in negated_indices:
                for j in range(k, k+len(sublist)):
                    preds.append(j)
            k += len(sublist)
                    
        return preds
    
    if output_style == "entities":
        return list(set(entity_indices) - set(negated_indices)), negated_indices
                    


def instead(transformed_sent, spacy_mapping, output_style="tags"):
    """Detects entities that are negated by the word "instead".
    `transformed_sent`: The first output of `transform_sentence` on the sentence represented as triples.
    `spacy_mapping`: The output of `get_spacy_tokens` given `transformed_sent` and the associated entity indices.
    `output_style`: If "tags", returns a series of True (negated) and False (non-negated), where each index is True if negated by "instead"."""
    
    string_sent = " ".join([item for sublist in transformed_sent for item in sublist])
    
    doc = nlp(string_sent)
    
    negated_indices = []
    
    for ent_idx, spacy_idx in spacy_mapping.items():
        i = spacy_idx
        root_hits = 0
        negated = False
        while root_hits != 2 and negated == False:  # i.e., while there is a head. In spacy, the main clause verb is its own head
            i = doc[i].head.i
            if i == doc[i].head.i:
                root_hits += 1
            if str(doc[i]) == "of" and str(doc[i-1]) == "instead":
                negated = True
                
        if negated:
            negated_indices.append(ent_idx)
            
        else:
            for i in range(6):
                if spacy_idx - i >= 0:
                    if doc[spacy_idx-i].text == "instead":
                        negated_indices.append(ent_idx)
                                                        
    return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)


def more_less_than(transformed_sent, spacy_mapping, output_style="tags"):
    """Detects entities that are negated by the words "more" or "less" and "than".
    `transformed_sent`: The first output of `transform_sentence` on the sentence represented as triples.
    `spacy_mapping`: The output of `get_spacy_tokens` given `transformed_sent` and the associated entity indices.
    `output_style`: If "tags", returns a series of True (negated) and False (non-negated), where each index is True if negated by "more/less" + "than"."""
    
    string_sent = " ".join([item for sublist in transformed_sent for item in sublist])
        
    negated_indices = []
    
    if "more than" not in string_sent.lower() and "less than" not in string_sent.lower():
        return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)
    
    for ent_idx, spacy_idx in spacy_mapping.items():
        i = spacy_idx
        negated = False
        if i == 0:
            continue
        if "more than" in " ".join(transformed_sent[ent_idx-1]).lower() or "less than" in " ".join(transformed_sent[ent_idx-1]).lower():
            negated = True
        if negated:
            negated_indices.append(ent_idx)
                                                        
    return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)
        


def without(transformed_sent, spacy_mapping, output_style="tags"):
    """Detects entities that are negated by the word "without".
    `transformed_sent`: The first output of `transform_sentence` on the sentence represented as triples.
    `spacy_mapping`: The output of `get_spacy_tokens` given `transformed_sent` and the associated entity indices.
    `output_style`: If "tags", returns a series of True (negated) and False (non-negated), where each index is True if negated by "without"."""
    
    string_sent = " ".join([item for sublist in transformed_sent for item in sublist])
    
    doc = nlp(string_sent)
        
    negated_indices = []
        
    prev_negated = False
    
    if "without" not in string_sent.lower():
        return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)
    
    for ent_idx, spacy_idx in spacy_mapping.items():
        
        i = spacy_idx
        negated = False
        if i == 0:
            continue
        
        if "without" in " ".join(transformed_sent[ent_idx-1]).lower() or prev_negated:
            negated = True
            if len(doc) > i+1:
                if doc[i+1].pos_ == "CCONJ":
                    prev_negated = True
                else:
                    prev_negated = False
                    
        if negated:
            negated_indices.append(ent_idx)
                                                        
    return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)


def not_(transformed_sent, spacy_mapping, output_style="tags"):
    """Detects entities that are negated by "no", "not", and cliticized versions of the latter.
    `transformed_sent`: The first output of `transform_sentence` on the sentence represented as triples.
    `spacy_mapping`: The output of `get_spacy_tokens` given `transformed_sent` and the associated entity indices.
    `output_style`: If "tags", returns a series of True (negated) and False (non-negated), where each index is True if negated by "no" or "not"."""
        
    string_sent = " ".join([item for sublist in transformed_sent for item in sublist])
    
    doc = nlp(string_sent)
        
    negated_indices = []
        
    entity_indices = sorted(spacy_mapping)
    
    end_punct = [".", "!", "?"]
    
    nots = {"not", "n't", "no", "don't", "doesn't", "aren't", "isn't"}
    
    for ent_idx, spacy_idx in spacy_mapping.items():
        
        if ent_idx in negated_indices:
            continue
        
        i = spacy_idx
            
        if ent_idx == 0 or spacy_idx == 0:
            continue
        
        negated = False
        
        negator = False
        
        if i == 0:
            continue
            
        if len(nots - set(" ".join(transformed_sent[ent_idx-1]).lower().split())) != len(nots):
            if "." not in " ".join(transformed_sent[ent_idx-1]) and             "!" not in " ".join(transformed_sent[ent_idx-1]) and             "?" not in " ".join(transformed_sent[ent_idx-1]):
                negator = True
            else:
                occurrences = [k for k, n in enumerate(transformed_sent[ent_idx-1]) if n[-1] in end_punct]
                if occurrences:
                    if len(nots - set(transformed_sent[ent_idx-1][occurrences[-1]:])) != len(nots):
                        negator = True
                
        elif not negated and len(nots - set(" ".join(transformed_sent[ent_idx]).lower().split())) != len(nots):
            for negator in nots:
                if negator in transformed_sent[ent_idx] and transformed_sent[ent_idx].index(negator) < i:
                    negator = True
                    break
                    
        if negator and "see" not in transformed_sent[ent_idx-1]:
            negated = True
            negated_indices.append(ent_idx)
            if i != len(doc)-1 and ent_idx != entity_indices[-1]:
                j = i+1
                ent_idx_idx = entity_indices.index(ent_idx)+1
                while j - i < 10 and j < len(doc) and ent_idx_idx < len(entity_indices):
                    if doc[j].lemma_ not in ["need", "want"] and doc[j].pos_ not in ["CCONJ", "ADJ", "NOUN", "DET", "PRON"]:
                        break
                    elif doc[j].text in transformed_sent[entity_indices[ent_idx_idx]][0]:
                        if entity_indices[ent_idx_idx] not in negated_indices:
                            negated_indices.append(entity_indices[ent_idx_idx])
                        ent_idx_idx += 1
                    j += 1
                
    return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)



def verbs(transformed_sent, spacy_mapping, output_style="tags"):
    """Detects entities that are negated by the verbs "replace" and "remove" (in passive voice).
    `transformed_sent`: The first output of `transform_sentence` on the sentence represented as triples.
    `spacy_mapping`: The output of `get_spacy_tokens` given `transformed_sent` and the associated entity indices.
    `output_style`: If "tags", returns a series of True (negated) and False (non-negated), where each index is True if negated by the above verbs."""
    
    string_sent = " ".join([item for sublist in transformed_sent for item in sublist])
    
    doc = nlp(string_sent)
    
    negated_indices = []
    
    for ent_idx, spacy_idx in spacy_mapping.items():
        i = spacy_idx
        root_hits = 0
        negated = False
        while root_hits != 2 and negated == False and doc[i].head.pos_ != "ADP":  # i.e., while there is a head. In spacy, the main clause verb is its own head
            i = doc[i].head.i
            if i == doc[i].head.i:
                root_hits += 1
            if doc[i].lemma_ in ["remove", "replace"]:
                negated = True
                
        if negated:
            negated_indices.append(ent_idx)
            
    return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)



def comparative(transformed_sent, spacy_mapping, output_style="tags"):
    """Detects entities that are negated by comparatives such as "bigger".
    `transformed_sent`: The first output of `transform_sentence` on the sentence represented as triples.
    `spacy_mapping`: The output of `get_spacy_tokens` given `transformed_sent` and the associated entity indices.
    `output_style`: If "tags", returns a series of True (negated) and False (non-negated), where each index is True if negated by a comparative."""
    
    string_sent = " ".join([item for sublist in transformed_sent for item in sublist])
    
    doc = nlp(string_sent)
        
    negated_indices = []
    
    comparative = False
    
    for ent_idx, spacy_idx in spacy_mapping.items():
        
        i = spacy_idx
        negated = False
        
        if comparative:
            negated = True
            
        comparative = False
        
        if doc[i].pos_ == "ADJ":
            if doc[i].text.endswith("er") and not doc[i].lemma_.endswith("er") and nlp(doc[i].lemma_)[0].pos_ == "ADJ":
                if len(doc) > i+1 and doc[i+1].text == "than":
                    comparative = True
                
        if negated:
            negated_indices.append(ent_idx)
        
    return get_negator_output(transformed_sent, spacy_mapping.keys(), negated_indices, output_style)


def predict_and_evaluate(sent_list, negators, output_type="tags", word_of_interest=None, verbose=False):
    """Given a list of sentences and `output_type`, simultaneously makes predictions and compares against the gold data."""

    preds = predict_sent_list(sent_list, negators, output_type, word_of_interest)
    gold = prepare_gold_data(sent_list, output_type, word_of_interest)
    
    evaluate(preds, gold, output_type, verbose)
    

negators = [instead, more_less_than, without, not_, comparative, verbs]


predict_and_evaluate(all_sents, negators, "entities", None, True)
predict_and_evaluate(all_sents, negators, "tags", None, True)
# print(all_sents)
# pred_neg = predict_one_sentence([("I","O", False), ("want","O", False), ("a","O", False), ("red","I-ATTRIBUTE", False), ("shirt","I-PRODUCT", False), ("but","O", False), ("not","O", False), ("blue","I-ATTRIBUTE", False), ("is","O", False), ("okay","O", False)], negators, "entities", None, True)

def predict(taggedTokens, outputStyle="entities"):
    pred_neg = predict_one_sentence(taggedTokens, negators, outputStyle, None, True)
    result = {"tokens":[], "tags":[], "isNegative":[]}
    if outputStyle =="tags":
        for tokentag, neg in zip(taggedTokens, pred_neg):
            result["tokens"].append(tokentag[0])
            result["tags"].append(tokentag[1])
            result["isNegative"].append(neg)
        return result
    elif outputStyle == "entities":
        entities = {"Desired":[], "Undesired":[]}
        entities["Desired"] = pred_neg[0]
        entities["Undesired"] = pred_neg[1]
        return entities
    else:
        return taggedTokens

# ### Reasons why scope detection is not always successful (based on partial sample of sentences):
# - No negator:
#   - instead (3)
#   - without (4)
#   - comparative (1) (e.g. "deeper than a regular-True tub")
#   - negation scope would include entities we don't want to be negated (2) (e.g., "counter top with no drill-True holes-True for the faucet-False")
#   - can X be removed (1)
#   - negative affix (1)
# - Miscellaneous:
#   - "I would like to know if the wool-like top side is soft as well, or if it is scratchy-True"
#   - "You have it for the gold-True and black-True but i don't [sic] want it for the navy"
#   - "I found a vanity top I love but it has only one-True hole-True for taps and that doesn't suit our needs"
#  
#  
# Not sure:
# - I'm looking for X, not Y or Z-True

# ## Ways of indicating negation (incomplete):
# 
# - "don't want (any)"
# - "no"
# - "no more than"
# - "less than"
# - "un-"
# - "without"
# - "only"
# - "-less"
# - "instead of"
# - "too" (what comes before is negated, e.g. in "pink is too light", "pink" is negated)
# - "not too" (what comes after is negated, e.g. in "not too light", "light" is negated)
# - "the website only gives the option of"
# - "without X or Y" (two separately annotated entities separated by conjunction)


if __name__ == "__main__":
    main()

