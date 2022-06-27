import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

color_data = pd.read_csv("data/colors.csv")

color_list = [color.replace("_", " ") for color in list(color_data["air_force_blue_raf"])]

color_list.append("green")
color_list.remove("linen")
color_list.remove("jet")
color_list.remove("quartz")

def tag_color(sent):
    """
    Uses a color list to identify colors in the input data.
    Colors can be multiple words long; scans with an appropriate window size for each color.
    Picks the color term comprising the most words, e.g. "apple green" over "green".
    Sent should be in the form of a list of tuples (token, IOB-tag).
    """
    just_words = [word.lower() for word, iob in sent]
    max_color_length = [0 for word in sent]
    
    for color in color_list:
        color = color.split()
        start_index = 0
        while start_index + len(color) <= len(sent):
            if just_words[start_index:start_index+len(color)] == color:
                for i in range(start_index, start_index+len(color)):
                    if len(color) > max_color_length[i]:
                        sent[i] = (sent[i][0], "I-COLOUR")
                        max_color_length[i] = len(color)
            start_index += 1
            
    return sent

def tag_size(sent):
    """
    Searches for patterns involving size.
    Sent should be in the form of a list of tuples (token, IOB-tag).
    """
    
    second_position = ["inch", "inches", "''", '""', '"', "'", "in", "feet", "cm", "mm"]
    
    third_position_adj = ["long", "wide", "high", "deep", "long", "tall"]
    
    third_position_noun = ["width" , "height", "depth", "length", "waist", "leg"]
    
    bed_sizes = ["king", "queen"]
    
    adjectives = ["bigger", "large", "small", "biggest", "largest", "smallest", "deepest", "widest", "highest", "tallest"]
    
    clothing_sizes = ["small", "medium", "large"]
    
    just_words = [word.lower() for word, iob in sent]
    
    for i, word in enumerate(just_words):
        if nlp(word)[0].pos_ == "NUM":
            if i < len(sent)-1:
                if just_words[i+1] in second_position:
                    sent[i] = (sent[i][0], "I-SIZE")
                    sent[i+1] = (sent[i+1][0], "I-SIZE")
                    if i < len(sent)-2:
                        if just_words[i+2] in third_position_adj or just_words[i+2] in third_position_noun:
                            sent[i+2] = (sent[i+2][0], "I-SIZE")
                        elif i < len(sent)-3:
                            if just_words[i+2].lower() == "in" and just_words[i+3] in third_position_noun:
                                sent[i+2] = (sent[i+2][0], "I-SIZE")
                                sent[i+3] = (sent[i+3][0], "I-SIZE")
        elif word in bed_sizes:
            sent[i] = (sent[i][0], "I-SIZE")
            if i < len(sent)-1:
                if just_words[i+1].lower() == "size":
                    sent[i+1] = (sent[i+1][0], "I-SIZE")
        
        elif word in adjectives:
            sent[i] = (sent[i][0], "I-SIZE")
            
        elif word.lower() == "size":
            if i < len(sent)-1:
                if nlp(just_words[i+1])[0].pos_ == "NUM" or just_words[i+1].lower() in clothing_sizes:
                    sent[i] = (sent[i][0], "I-SIZE")
                    sent[i+1] = (sent[i+1][0], "I-SIZE")
    
    return sent

def tag_shape(sent):
    """
    Searches for single words involving size.
    Sent should be in the form of a list of tuples (token, IOB-tag).
    """
    shapes = ["circle", "circular", "rectangle", "rectangular", "square", "oval", "round"]
    just_words = [word.lower() for word, iob in sent]
    for i, word in enumerate(just_words):
        if word.lower() in shapes:
            sent[i] = (sent[i][0], "I-SHAPE")
            
    return sent

def tag_price(sent):
    """
    Searches for patterns involving price.
    Sent should be in the form of a list of tuples (token, IOB-tag).
    """
    just_words = [word for word, iob in sent]
    spacy_sent_string = ""
    for word in just_words:
        if word.startswith("'"):
            spacy_sent_string = spacy_sent_string + word
        else:
            spacy_sent_string = spacy_sent_string + " " + word
    spacy_sent_string = spacy_sent_string[1:]
    spacy_sent = nlp(spacy_sent_string)
    for i, word in enumerate(spacy_sent):
        if "price" in word.text or "expensive" in word.text:
            sent[i] = (sent[i][0], "I-PRICE")
        elif ("price" in word.head.text or "expensive" in word.head.text) and word.pos_ in ["ADJ", "ADV", "SYM", "CCONJ"]:
            sent[i] = (sent[i][0], "I-PRICE")
        elif word.pos_ == "NUM" and word.head.text == "for":
            sent[i] = (sent[i][0], "I-PRICE")
            amount_index = word.i
            for word2 in spacy_sent:
                if word2.head.i == amount_index and word2.pos_ in ["ADJ", "ADV", "SYM", "CCONJ"]:
                    sent[word2.i] = (sent[word2.i][0], "I-PRICE")
            
    return sent