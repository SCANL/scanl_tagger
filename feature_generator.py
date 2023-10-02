import time, nltk
import gensim.downloader as api
from gensim.models import KeyedVectors as word2vec
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from spellchecker import SpellChecker

spell = SpellChecker()

vector_size = 128

#Conjunctions and determiners are closed set words, so we can soft-code them by doing a lookup on their
#Word embeddings. This avoids the problem with hard-coding (i.e., assuming the word is always a closet set word)
#while still giving our approach the ability to determine if we're in the most-likely context of them being a closed set word
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
    data = createPrepositionFeature(data)
    data = firstWordLength(data)
    data = firstWordCaps(data)
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
    identifiers = data["IDENTIFIER"]
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
    modelGensimTokens = word2vec.load_word2vec_format('../code2vec/token_vecs.txt', binary=False)
    modelGensimMethods = word2vec.load_word2vec_format('../code2vec/target_vecs.txt', binary=False)

    return modelGensimTokens, modelGensimMethods, modelGensimEnglish