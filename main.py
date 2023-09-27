import os
import sqlite3
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import classifier_multiclass
import gensim.downloader as api
from gensim.models import KeyedVectors as word2vec
import random
import nltk
from scipy.spatial.distance import cosine

from spellchecker import SpellChecker
spell = SpellChecker()

# import classifier_training_set_generator
nltk.download('universal_tagset')

input_file = 'input/det_conj_db2.db'
sql_statement = 'select * from base'
# sql_statement = 'select * from training_set_conj_other order by random()';
# sql_statement = 'select * from training_set_norm order by random()';
# sql_statement = 'select * from training_set_norm_other order by random()';
identifier_column = "ID"
# independent_variables = ['WORD', 'POSITION', 'MAXPOSITION', 'NORMALIZED_POSITION', 'CONTEXT']
# independent_variables = ['TYPE', 'WORD', 'SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG', 'NORMALIZED_POSITION', 'CONTEXT']
independent_variables_base = ['NORMALIZED_POSITION']
dependent_variable = 'CORRECT_TAG'
vector_size = 128
vector_size_e = 300

# Training Seed: 2797879, 532479
# Classifier seed: 1271197, 948572

#db2
# Training Seed: 2227339
# Classifier Seed: 3801578
# SEED: 1340345

seed = 1340345
print("SEED: " + str(seed))
trainingSeed = 2227339
classifierSeed = 3801578
np.random.seed(1129175)
random.seed(seed)

#Conjunctions and determiners are closed set words, so we can soft-code them by doing a lookup on their
#Word embeddings. This avoids the problem with hard-coding (i.e., assuming the word is always a closet set word)
#while still giving our approach the ability to determine if we're in the most-likely context of them being a closed set word
#"as if", "as long as", "as much as", "as soon as", "as far as", "as though", "by the time", "in as much as", "in as much", "in order to", "in order that", "in case",
#"now that" "now since", "now when" "even if" "even though" "provided that" "if then", "if when", "if only",
# "just as" "where if" "or not" "neither nor" "not only but also",  "whether or" "provided that", "as well as"
# "as well as"
conjunctions = {"for", "and", "nor", "but", "or", "yet", "so", "although", "after", "before", "because", "how",
                "if", "once", "since", "until", "unless", "when", "as", "that", "though", "till", "while", "where", "after",
                "although", "as", "lest", "though", "now", "even", "provided", "else", "where", "wherever", "whereas", 
                "whether", "since", "because", "whose", "whoever", "unless", "while", "before", "why", "so that", "until", 
                "how", "since", "than", "till", "whenever", "supposing", "when", "what", "also", "otherwise", "for", "and",  "nor", "but", 
                "so that", "or", "such that", "yet", "as soon as", "so", "also", "whoever", "yet", "while", "still", "until", "too", "unless", 
                "only", "since", "however", "as if", "no less than", "no less than", "which", "otherwise", "where", "in order that", 
                "who", "than", "after", "as", "because", "either or", "whoever", "nevertheless", "though", "else", "although", "if", 
                "while", "till"}

determiners = {"a", "all", "an", "another", "any", "anybody", "anyone", "anything", "anywhere", "both", "certain", "each", 
               "either", "enough", "every", "everybody", "everyone", "everything", "everywhere", "few", "fewer", "fewest", "last", "least", "less", 
               "little", "many", "more", "most", "much", "neither", "next", "no", "no one", "nobody", "none", "nothing", "nowhere", "once", 
               "said", "several", "some", "somebody", "something", "somewhere", "sufficient", "that", "the", "these", "this", "those", "us", 
               "various", "we", "what", "whatever", "which", "whichever", "you"}

prepositions = {"after", "although", "as", "at", "because", "before", "beside", "besides", "between", "by", "considering", "despite", "except", 
                "for", "from", "given", "granted", "if", "into", "lest", "like", "notwithstanding", "now", "of", "on", "once", "provided", "providing",
                "save", "seeing", "since", "so", "supposing", "than", "though", "till", "to", "unless", "until", "upon", "when", "whenever", "where",
                "whereas", "wherever", "while", "whilst", "with", "without"}

verbs = {'be','have','do','say','get','make','go','see','know','take','think','come','give','look','use','find','want','tell','put','mean','become','leave','work','need','feel','seem',
        'ask','show','try','call','keep','provide','hold','turn','follow','begin','bring','like','going','help','start','run','write','set','move','play','pay','hear','include',
        'believe','allow','meet','lead','live','stand','happen','carry','talk','appear','produce','sit','offer','consider','expect','suggest','let','read','require','continue',
        'lose','add','change','fall','remain','remember','buy','speak','stop','send','receive','decide','win','understand','describe','develop','agree','open','reach','build',
        'involve','spend','return','draw','die','hope','create','walk','sell','wait','cause','pass','lie','accept','watch','raise','base','apply','break','explain','learn',
        'increase','cover','grow','claim','report','support','cut','form','stay','contain','reduce','establish','join','wish','achieve','seek','choose','deal','face','fail',
        'serve','end','kill','occur','drive','represent','rise','discuss','love','pick','place','argue','prove','wear','catch','enjoy','eat','introduce','enter','present','arrive',
        'ensure','point','plan','pull','refer','act','relate','affect','close','identify','manage','thank','compare','announce','obtain','note','forget','indicate','wonder','maintain',
        'publish','suffer','avoid','express','suppose','finish','determine','design','listen','save','tend','treat','control','share','remove','throw','visit','exist','encourage',
        'force','reflect','admit','assume','smile','prepare','replace','fill','improve','mention','fight','intend','miss','discover','drop','hit','push','prevent','refuse','regard',
        'lay','reveal','teach','answer','operate','state','depend','enable','record','check','complete','cost','sound','laugh','realise','extend','arise','notice','define','examine',
        'fit','study','bear','hang','recognise','shake','sign','attend','fly','gain','perform','result','travel','adopt','confirm','protect','demand','stare','imagine','attempt','beat',
        'born','associate','care','marry','collect','voice','employ','issue','release','emerge','mind','aim','deny','mark','shoot','appoint','order','supply','drink','observe','reply','ignore',
        'link','propose','ring','settle','strike','press','respond','arrange','survive','concentrate','lift','approach','cross','test','charge','experience','touch','acquire','commit',
        'demonstrate','grant','prefer','repeat','sleep','threaten','feed','insist','launch','limit','promote','deliver','measure','own','retain','assess','attract','belong','consist',
        'contribute','hide','promise','reject','cry','impose','invite','sing','vary','warn','address','declare','destroy','worry','divide','head','name','stick','nod','recognize','train',
        'attack','clear','combine','handle','influence','realize','recommend','shout','spread','undertake','account','select','climb','contact','recall','secure','step','transfer','welcome',
        'conclude','disappear','display','dress','illustrate','imply','organise','direct','escape','generate','investigate','remind','advise','afford','earn','hand','inform','rely','succeed',
        'approve','burn','fear','vote','conduct','cope','derive','elect','gather','jump','last','match','matter','persuade','ride','shut','blow','estimate','recover','score','slip','count','hate',
        'attach','exercise','house','lean','roll','wash','accompany','accuse','bind','explore','judge','rest','steal','comment','exclude','focus','hurt','stretch','withdraw','back','fix','justify',
        'knock','pursue','switch','appreciate','benefit','lack','list','occupy','permit','surround','abandon','blame','complain','connect','construct','dominate','engage','paint','quote','view',
        'acknowledge','dismiss','incorporate','interpret','proceed','search','separate','stress','alter','analyse','arrest','bother','defend','expand','implement','possess','review','suit',
        'tie','assist','calculate','glance','mix','question','resolve','rule','suspect','wake','appeal','challenge','clean','damage','guess','reckon','restore','restrict','specify','constitute',
        'convert','distinguish','submit','trust','urge','feature','land','locate','predict','preserve','solve','sort','struggle','cast','cook','dance','invest','lock','owe','pour','shift','kick','kiss',
        'light','purchase','race','retire','bend','breathe','celebrate','date','fire','monitor','print','register','resist','behave','comprise','decline','detect','finance','organize','overcome',
        'range','swing','differ','drag','guarantee','oppose','pack','pause','relax','resign','rush','store','waste','compete','expose','found','install','mount','negotiate','sink','split','whisper','assure',
        'award','borrow','bury','capture','deserve','distribute','doubt','enhance','phone','sweep','tackle','advance','cease','concern','emphasise','exceed','qualify','slide','strengthen','transform',
        'favour','grab','lend','participate','perceive','pose','practise','satisfy','scream','smoke','sustain','tear','adapt','adjust','ban','consult','dig','dry','highlight','outline','reinforce','shrug',
        'snap','absorb','amount','block','confine','delay','encounter','entitle','plant','pretend','request','rid','sail','trace','trade','wave','cite','dream','flow','fulfil','lower','process','react','seize',
        'allocate','burst','communicate','defeat','double','exploit','fund','govern','hurry','injure','pray','protest','sigh','smell','stir','swim','undergo','wander','anticipate','collapse',
        'compose','confront','ease','eliminate','evaluate','grin','interview','remark','suspend','weigh','wipe','wrap','attribute','balance','bet','bound','cancel','condemn','convince',
        'correspond','dare','devise','free','gaze','guide','inspire','modify','murder','prompt','reverse','rub','slow','spot','swear','telephone','wind','admire','bite','crash','disturb','greet',
        'hesitate','induce','integrate','knit','line','load','murmur','render','shine','swallow','tap','translate','yield','accommodate','age','assert','await','book','brush','chase','comply',
        'copy','criticise','devote','evolve','flee','forgive','initiate','interrupt','leap','mutter','overlook','risk','shape','spell','squeeze','trap','undermine','witness','beg','drift',
        'echo','emphasize','enforce','exchange','fade','float','freeze','hire','in','object','pop','provoke','recruit','research','sense','situate','stimulate','abolish','administer','allege',
        'command','consume','convey','correct','educate','equip','execute','fetch','frown','invent','march','park','progress','reserve','respect','twist','unite','value','assign','cater','concede',
        'conceive','disclose','envisage','exhibit','export','extract','fancy','inherit','insert','instruct','interfere','isolate','opt','peer','persist','plead','price','regret','regulate','repair',
        'resemble','resume','speed','spin','spring','update','advocate','assemble','boost','breed','cling','commission','conceal','contemplate','criticize','decorate','descend','drain','edit',
        'embrace','excuse','explode','facilitate','flash','fold','function','grasp','incur','intervene','label','please','rescue','strip','tip','upset','advertise','aid','centre','classify',
        'coincide','confess','contract','crack','creep','decrease','deem','dispose','dissolve','dump','endorse','formulate','import','impress','market','reproduce','scatter','schedule','ship',
        'shop','spare','sponsor','stage','suck','sue','tempt','vanish','access','commence','contrast','depict','discharge','draft','enclose','enquire','erect','file','halt','hunt','inspect','omit',
        'originate','praise','precede','relieve','reward','round','seal','signal','smash','spoil','subject','target','taste','tighten','top','tremble','tuck','warm','activate','amend','arouse','bang',
        'bid','bow','campaign','characterise','circulate','clarify','compensate','compile','cool','couple','depart','deprive','desire','diminish','drown','embark','entail','entertain','figure',
        'fling','guard','manufacture','melt','neglect','plunge','project','rain','reassure','rent','revive','sentence','shed','slam','spill','stem','sum','summon','supplement','suppress','surprise',
        'tax','thrust','tour','transmit','transport','weaken','widen','bounce','calm','characterize','chat','clutch','confer','conform','confuse','convict','counter','debate','dedicate','dictate',
        'disagree','effect','flood','forbid','grip','heat','long','manipulate','merge','part','pin','position','prescribe','proclaim','punish','rebuild','regain','sack','strain','stroke','substitute',
        'supervise','term','time','toss','underline','abuse','accumulate','alert','arm','attain','boast','boil','carve','cheer','colour','compel','crawl','crush','curl','deposit','differentiate',
        'dip','dislike','divert','embody','exert','exhaust','fine','frighten','fuck','gasp','honour','inhibit','motivate','multiply','narrow','obey','penetrate','picture','presume','prevail',
        'pronounce','rate','renew','revise','rip','scan','scratch','shiver'}

hungarian = {'a', 'b', 'c', 'cb', 'cr', 'cx', 'dw', 'f', 'fn', 'g', 'h', 'i', 'l', 'lp', 'm', 'n', 'p', 's', 'sz', 'tm', 'u', 'ul', 'w', 'x', 'y'}
#["LAST_LETTER", 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE','DIGITS', 'CONJUNCTION', 'PREPOSITION', 'DETERMINER', 'METHODV_SCORE', 'METHODN_SCORE']
independent_variables_add = [[]]
independent_variables_add[0] += ["LAST_LETTER", 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'ENGLISHN_SCORE','METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE', 'FIRST_WORD_LENGTH', 'FIRST_WORD_CAPS'] # 'CONJUNCTION', 'DIGITS'
#independent_variables_add[0] += ["LAST_LETTER", 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'METHODN_SCORE', 'METHODV_SCORE', 'DETERMINER', 'POSITION', 'FREQUENCY', 'ENGLISHN_SCORE', 'ENGLISHV_SCORE', 'WORD LENGTH', 'TOKEN_SCORE','DIGITS', 'CONJUNCTION', 'PREPOSITION']
#["LAST_LETTER", 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'METHODV_SCORE', 'DETERMINER', 'POSITION']
# for i in range(0, vector_size_e):
#     independent_variables_add[0].append("VEC" + str(i))
# for i in range(0, vector_size):
#     independent_variables_add[0].append("MVEC" + str(i))

def createFeatures(data):
    startTime = time.time()
    modelTokens, modelMethods, modelGensimEnglish = createModel()
    data = createVerbVectorFeature(data, modelGensimEnglish)
    data = createDeterminerVectorFeature(data, modelGensimEnglish)
    data = createConjunctionVectorFeature(data, modelGensimEnglish)
    data = createPrepositionVectorFeature(data, modelGensimEnglish)
    data = createPreambleVectorFeature("CODE", data, modelTokens)
    data = createPreambleVectorFeature("METHOD", data, modelMethods)
    data = createPreambleVectorFeature("ENGLISH", data, modelGensimEnglish)
    data = createLetterFeature(data)
    data = maxPosition(data)
    data = wordPosTag(data)
    data = createSimilarityToVerbFeature("METHODV", modelMethods, data)
    data = createSimilarityToVerbFeature("ENGLISHV", modelGensimEnglish, data)
    data = createSimilarityToNounFeature("METHODN", modelMethods, data)
    data = createSimilarityToNounFeature("ENGLISHN", modelGensimEnglish, data)
    data = createDeterminerFeature(data)
    data = createDigitFeature(data)
    #data = createConjunctionFeature(data)
    data = createPrepositionFeature(data)
    data = firstWordLength(data)
    data = firstWordCaps(data)
    # data = createSimilarityToVerbFeature("TOKEN", modelTokens, data)
    # data = createWordVectorsFeature(modelGensimEnglish, data, "ENG")
    # data = createMethodWordVectorsFeature(modelMethods, data)
    # data = createFrequencyFeature(data)
    # data = createVowelFeature(data)
    print("Total Feature Time: " + str((time.time() - startTime)))
    return data

universal_to_custom = {
    'VERB': 'VERB',
    'NOUN': 'NOUN',
    'PROPN': 'NOUN',
    'ADJ': 'ADJ',
    'ADV': 'ADV',
    'ADP': 'ADP',
    'CCONJ': 'CONJ',
    'CONJ': 'CONJ',
    'SCONJ' : 'CONJ',
    'PRON' : 'DET',
    'SYM' : 'NM',
    'DET': 'DET',
    'NUM': 'NUM',
    'PRT': 'NM',
    'INTJ' : 'NM',
    'X': 'NM',
    '.': '.',
}

def wordPosTag(data):
    words = data["WORD"]
    word_tags = [universal_to_custom[nltk.pos_tag([word.lower()], tagset='universal')[-1][-1]] for word in words]
    pos_tags = pd.DataFrame(word_tags)
    pos_tags.columns = ['NLTK_POS']
    data = pd.concat([data, pos_tags], axis=1)
    return data


def firstWordLength(data):
    words = data["IDENTIFIER"]
    wordLengths = []

    for identifier in words:
        # Split the identifier into words using spaces as the delimiter
        word_list = identifier.split()

        if word_list:
            # Get the first word from the split identifier
            first_word = word_list[0]

            # Count the number of letters in the first word
            letters_count = sum(1 for char in first_word if char.isalnum())
        else:
            # Handle the case where the identifier is empty or doesn't contain any words
            letters_count = 0

        # Add the count to the wordLengths list
        wordLengths.append(letters_count)

    # Add the wordLengths list as a new column 'FIRST WORD LENGTH' to the 'data' DataFrame
    data['FIRST_WORD_LENGTH'] = wordLengths

    return data


def firstWordCaps(data):
    words = data["IDENTIFIER"]
    wordLengths = []

    for identifier in words:
        # Split the identifier into words using spaces as the delimiter
        word_list = identifier.split()

        if word_list:
            # Get the first word from the split identifier
            first_word = word_list[0]

            # Count the number of capital letters in the first word
            caps_count = sum(1 for char in first_word if char.isupper())
            caps_count = caps_count/len(first_word)
        else:
            # Handle the case where the identifier is empty or doesn't contain any words
            caps_count = 0

        # Add the count to the wordLengths list
        wordLengths.append(caps_count)

    # Add the wordLengths list as a new column 'FIRST WORD CAPS' to the 'data' DataFrame
    data['FIRST_WORD_CAPS'] = wordLengths

    return data

def maxPosition(data):
    identifiers = data["GRAMMAR_PATTERN"]
    maxPosition = pd.DataFrame([len(identifier.split()) for identifier in identifiers])
    maxPosition.columns = ['MAXPOSITION']
    data = pd.concat([data, maxPosition], axis=1)
    return data

def createFrequencyFeature(data):
    words = data["WORD"]
    frequency = {}
    for word in words:
        word = word.lower()
        if word in frequency:
            frequency[word] = frequency[word] + 1
        else:
            frequency[word] = 1
    frequencyList = pd.DataFrame([frequency[word.lower()] for word in words])
    frequencyList.columns = ['FREQUENCY']
    data = pd.concat([data, frequencyList], axis=1)
    return data

def count_vowels(word):
    # Convert the word to lowercase to make the function case-insensitive
    word = word.lower()

    # Define a set of vowels
    vowels = {'a', 'e', 'i', 'o', 'u'}

    # Initialize a variable to store the count of vowels
    vowel_count = 0

    word_size = len(word)

    # Iterate through each character in the word
    for char in word:
        # Check if the character is a vowel
        if char in vowels:
            vowel_count += 1

    return vowel_count

def createVowelFeature(data):
    words = data["WORD"]
    isVowelorConsonant = pd.DataFrame([count_vowels(word) for word in words])
    isVowelorConsonant.columns = ["VOWELCOUNT"]
    data = pd.concat([data, isVowelorConsonant], axis=1)
    return data

def average_word_vectors(word_set, word2vec_model):
    word_vectors = []
    for word in word_set:
        if word in word2vec_model.index_to_key:
            word_vectors.append(word2vec_model.get_vector(word))

    if not word_vectors:
        raise ValueError("None of the words in the set exist in the Word2Vec model.")

    return np.mean(word_vectors, axis=0)

def compute_similarity(verb_vector, target_word, model):
    # Compute the cosine similarity between the two vectors
    try:
        target_word_vector = model.get_vector(key=target_word, norm=True)
        similarity = 1 - cosine(verb_vector, target_word_vector)
        return similarity
    except KeyError:
        return 0.0

def createVerbVectorFeature(data, model):
    words = data["WORD"]
    vector = average_word_vectors(verbs, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['VERB_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createDeterminerVectorFeature(data, model):
    words = data["WORD"]
    vector = average_word_vectors(conjunctions, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['DET_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createPrepositionVectorFeature(data, model):
    words = data["WORD"]
    vector = average_word_vectors(prepositions, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['PREP_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createConjunctionVectorFeature(data, model):
    words = data["WORD"]
    vector = average_word_vectors(conjunctions, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['CONJ_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createPreambleVectorFeature(name, data, model):
    words = data["WORD"]
    vector = average_word_vectors(hungarian, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = [name+'PRE_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createPrepositionFeature(data):
    words = data["WORD"]
    isPreposition = pd.DataFrame([1 if word.lower() in prepositions else 0 for word in words])
    isPreposition.columns = ["PREPOSITION"]
    data = pd.concat([data, isPreposition], axis=1)
    return data

def createConjunctionFeature(data):
    words = data["WORD"]
    isConjunction = pd.DataFrame([1 if word.lower() in conjunctions else 0 for word in words])
    isConjunction.columns = ["CONJUNCTION"]
    data = pd.concat([data, isConjunction], axis=1)
    return data


def createDeterminerFeature(data):
    words = data["WORD"]
    isDeterminer = pd.DataFrame([1 if word.lower() in determiners else 0 for word in words])
    isDeterminer.columns = ["DETERMINER"]
    data = pd.concat([data, isDeterminer], axis=1)
    return data


def createDigitFeature(data):
    words = data["WORD"]
    isDigits = pd.DataFrame([1 if word.isdigit() else 0 for word in words])
    isDigits.columns = ["DIGITS"]
    data = pd.concat([data, isDigits], axis=1)
    return data


def createLetterFeature(data):
    lastLetters = pd.DataFrame(np.array([ord(word[len(word) - 1].lower()) for word in data["WORD"]]))
    lastLetters.columns = ["LAST_LETTER"]
    data = pd.concat([data, lastLetters], axis=1)
    return data

def get_word_vector(word, model, vector_size):
    try:
        # Try to get the word vector from the model
        vector = model.get_vector(word)
        return vector
    except KeyError:
        # If the word is not in the model, correct it using pyspellchecker
        corrected_word = spell.correction(word)
        
        try:
            # Try again to get the word vector for the corrected word
            vector = model.get_vector(corrected_word)
            return vector
        except KeyError:
            # If the corrected word is still not in the model, return a zero vector
            return np.zeros(model.vector_size)

#Get word vectors for our closed set words
def createWordVectorsFeature(model, data, name="DEFAULT"):
    words = data["WORD"]
    vectors = None
    if name == "ENG":
        vectors = [get_word_vector(word.lower(), model, 300) for word in words]
    else:
        vectors = [get_word_vector(word.lower(), model, 128) for word in words]
    cnames = [f'VEC{i}' for i in range(0, len(vectors[0]))]
    df = pd.DataFrame()
    for i in range(0, len(vectors[0])):
        df = pd.concat([df, pd.DataFrame([vector[i] for vector in vectors])], axis=1)
    df.columns = cnames

    data = pd.concat([data, df], axis=1)
    return data

def get_word_similarity(word, word2, model):
    try:
        # Try to get the word vector from the model
        vector = model.similarity(word, word2)
        return vector
    except KeyError:
        return 0

def createSimilarityToVerbFeature(name, model, data):
    words = data["WORD"]
    scores = pd.DataFrame([get_word_similarity("verb", word.lower(), model) for word in words])
    scores.columns = [name+'_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createSimilarityToNounFeature(name, model, data):
    words = data["WORD"]
    scores = pd.DataFrame([get_word_similarity("noun", word.lower(), model) for word in words])
    scores.columns = [name+'_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createMethodWordVectorsFeature(model, data):
    words = data["WORD"]
    vectors = [get_word_vector(word.lower(), model, 128) for word in words]
    cnames = [f'MVEC{i}' for i in range(0, vector_size)]
    df = pd.DataFrame()
    for i in range(0, vector_size):
        df = pd.concat([df, pd.DataFrame([vector[i] for vector in vectors])], axis=1)
    df.columns = cnames

    data = pd.concat([data, df], axis=1)
    return data

def createModel(pklFile=""):
    modelGensimEnglish = api.load('fasttext-wiki-news-subwords-300')
    modelGensimTokens = word2vec.load_word2vec_format('./code2vec/token_vecs.txt', binary=False)
    modelGensimMethods = word2vec.load_word2vec_format('./code2vec/target_vecs.txt', binary=False)

    return modelGensimTokens, modelGensimMethods, modelGensimEnglish

def read_input(sql, conn):
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")

    input_data_copy = input_data.copy()
    rows = input_data_copy.values.tolist()
    random.shuffle(rows)
    shuffled_input_data = pd.DataFrame(rows, columns=input_data.columns)

    input_data = createFeatures(shuffled_input_data)
    return input_data

def main():
    count = 0
    for feature_list in independent_variables_add:
        count = count + 1
        start = time.time()
        intervalStart = start

        # ###############################################################
        print(" --  -- Started: Reading Database --  -- ")
        connection = sqlite3.connect(input_file)
        df_input = read_input(sql_statement, connection)
        print(" --  -- Completed: Reading Input --  -- ")
        # ###############################################################

        category_variables = []
        text_column = ""

        feature_list = independent_variables_base + feature_list
        df_input.set_index(identifier_column, inplace=True)
        df_features = df_input[feature_list]
        if 'NLTK_POS' in feature_list:
            category_variables.append('NLTK_POS')
            df_features['NLTK_POS'] = df_features['NLTK_POS'].astype(str)
        if 'TYPE' in feature_list:
            category_variables.append('TYPE')
            df_features['TYPE'] = df_features['TYPE'].astype(str)

        df_class = df_input[[dependent_variable]]

        if not os.path.exists('output'):
            os.makedirs('output')
        filename = 'output/results.txt'
        if os.path.exists(filename):
            append_write = 'a'
        else:
            append_write = 'w'

        results_text_file = open(filename, append_write)
        results_text_file.write(datetime.now().strftime("%H:%M:%S") + "\n")
        for category_column in category_variables:
            if category_column in df_features.columns:
                df_features[category_column] = df_features[category_column].astype('category')
                d = dict(enumerate(df_features[category_column].cat.categories))
                results_text_file.write(str(category_column) + ":" + str(d) + "\n")
                df_features[category_column] = df_features[category_column].cat.codes

        print(" --  -- Distribution of labels in corpus --  -- ")
        print(df_class[dependent_variable].value_counts())

        results_text_file.write("SQL: %s\n" % sql_statement)
        results_text_file.write("Features: {number}. {features}\n".format(features=df_features, number=count))
        algorithms = [classifier_multiclass.Algorithm.RANDOM_FOREST]
        for index in range(1):
            classifier_multiclass.perform_classification(df_features, df_class, text_column, results_text_file,
                                                         'output',
                                                         algorithms, trainingSeed, classifierSeed)
            print("Run #" + str(index))
            print("Time Stamp: " + str(time.time() - intervalStart))
            print("Training Seed: " + str(trainingSeed))
            print("Classifier seed: " + str(classifierSeed))
            intervalStart = time.time()

        end = time.time()
        print("Process completed in " + str(end - start) + " seconds")


############CURRENTLY NOT EXECUTED###############

def annotate_word(params):
    input_model = 'output/model_RandomForestClassifier.pkl'

    data = {
        'NORMALIZED_POSITION': params['normalized_length'],
        'LAST_LETTER': params['last_letter'],
        'CONTEXT': params['code_context'],
        'MAXPOSITION': params['max_position'],
        'NLTK_POS': params['nltk_pos'],
        'POSITION': params['position'],
        'VERB_SCORE': params['verb_score'],
        'DET_SCORE': params['det_score'],
        'PREP_SCORE': params['prep_score'],
        'CONJ_SCORE': params['conj_score'],
        'PREPOSITION': params['prep'],
        'DETERMINER': params['det'],
        'ENGLISHV_SCORE': params['englishv_score'],
        'ENGLISHN_SCORE': params['englishn_score'],
        'METHODN_SCORE': params['methodn_score'],
        'METHODV_SCORE': params['methodv_score'],
        'CODEPRE_SCORE': params['codepre_score'],
        'METHODPRE_SCORE': params['methodpre_score'],
        'ENGLISHPRE_SCORE': params['englishpre_score'],
        'FIRST_WORD_LENGTH': params['first_word_len'],
        'FIRST_WORD_CAPS': params['first_word_caps'],
    }

    df_features = pd.DataFrame(data, columns=independent_variables_base + independent_variables_add[0])

    clf = joblib.load(input_model)
    y_pred = clf.predict(df_features)
    return y_pred[0]


def read_from_database():
    input_file = 'input/revision_testing_db.db'
    sql_statement = "select * from testing_set_cp_minor"
    # sql_statement = "select * from testing_set_ca_minor"
    # sql_statement = "select * from testing_set_np_minor"
    # sql_statement = "select * from testing_set_na_minor"
    connection = sqlite3.connect(input_file)

    df_input = pd.read_sql_query(sql_statement, connection)
    outputFile = "output/model_DecisionTreeClassifier_predictions.csv"
    print(" --  --  --  -- Read " + str(len(df_input)) + " input rows --  --  --  -- ")
    print("IDENTIFIER,GRAMMAR_PATTERN,WORD,SWUM,STANFORD,CORRECT,PREDICTION,MATCH,SYSTEM,CONTEXT,IDENT",
          file=open(outputFile, "a"))
    df_input = createFeatures(df_input)
    
    category_variables = []
    if 'NLTK_POS' in df_input:
        category_variables.append('NLTK_POS')
        df_input['NLTK_POS'] = df_input['NLTK_POS'].astype(str)
    
    for category_column in category_variables:
        if category_column in df_input.columns:
            df_input[category_column] = df_input[category_column].astype('category')
            d = dict(enumerate(df_input[category_column].cat.categories))
            df_input[category_column] = df_input[category_column].cat.codes
    
    results_list = []
    start = time.time()
    for i, row in df_input.iterrows():
        actual_word = row['WORD']
        actual_identifier = row['IDENTIFIER']
        actual_pattern = row['GRAMMAR_PATTERN']
        
        params = {
            'normalized_length': row['NORMALIZED_POSITION'],
            'code_context': row['CONTEXT'],
            'last_letter': row['LAST_LETTER'],
            'max_position': row['MAXPOSITION'],
            'position': row['POSITION'],
            'determiner': row['DETERMINER'],
            'nltk_pos' : row['NLTK_POS'],
            #'conjunction': row['CONJUNCTION'],
            'verb_score': row['VERB_SCORE'],
            'det_score': row['DET_SCORE'],
            'prep_score': row['PREP_SCORE'],
            'conj_score': row['CONJ_SCORE'],
            'prep': row['PREPOSITION'],
            'det': row['DETERMINER'],
            'englishv_score': row['ENGLISHV_SCORE'],
            'englishn_score': row['ENGLISHN_SCORE'],
            'methodn_score': row['METHODN_SCORE'],
            'methodv_score': row['METHODV_SCORE'],
            'codepre_score': row['CODEPRE_SCORE'],
            'methodpre_score': row['METHODPRE_SCORE'],
            'englishpre_score': row['ENGLISHPRE_SCORE'],
            'first_word_len': row['FIRST_WORD_LENGTH'],
            'first_word_caps': row['FIRST_WORD_CAPS']
        }

        
        result = annotate_word(params)

        # Append the results to the results_list
        results_list.append({
            'identifier': actual_identifier,
            'pattern': actual_pattern,
            'word': actual_word,
            'correct': row['CORRECT_TAG'],
            'prediction': result,
            'agreement': (row['CORRECT_TAG'] == result),
            'system_name': row['SYSTEM'],
            'context': row['CONTEXT'],
            'ident': row['IDENTIFIER_CODE'],
            'last_letter': row['LAST_LETTER'],
            'max_position': row['MAXPOSITION'],
            'position': row['POSITION']
        })

    end = time.time()
    print("Process completed in " + str(end - start) + " seconds")

    results_df = pd.DataFrame(results_list)
    output_file = "output/model_RandomForestClassifier_predictions.csv"
    results_df.to_csv(output_file, index=False, mode='a')



if __name__ == "__main__":
    #read_from_database()
    main()
