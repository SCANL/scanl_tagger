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
    """
    Create various features for the input data using pre-trained Word2Vec models and other techniques.

    This function adds multiple features to the input DataFrame 'data' based on pre-trained Word2Vec models and other
    techniques. These features include vector similarity scores, word embeddings, and linguistic properties.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column and other relevant columns.

    Returns:
        pandas.DataFrame: The input DataFrame with additional features added.
    """
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
    """
    Perform part-of-speech tagging on words in the 'WORD' column of the DataFrame.

    This function uses NLTK's part-of-speech tagging to tag each word in the 'WORD' column of the input DataFrame with
    custom POS tags (assuming a mapping from universal tags to custom tags). The tagged POS information is added as a new
    column 'NLTK_POS' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'NLTK_POS' column containing custom POS tags.
    """
    words = data["WORD"]
    word_tags = [universal_to_custom[nltk.pos_tag([word.lower()], tagset='universal')[-1][-1]] for word in words]
    pos_tags = pd.DataFrame(word_tags)
    pos_tags.columns = ['NLTK_POS']
    data = pd.concat([data, pos_tags], axis=1)
    return data


def maxPosition(data):
    """
    Calculate and add a 'MAXPOSITION' column to the DataFrame indicating the maximum number of words in each identifier.

    This function calculates the maximum number of words (based on spaces as delimiters) in each identifier in the 'IDENTIFIER'
    column of the input DataFrame and adds this information as a new column 'MAXPOSITION' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing an 'IDENTIFIER' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'MAXPOSITION' column.
    """
    identifiers = data["IDENTIFIER"]
    maxPosition = pd.DataFrame([len(identifier.split()) for identifier in identifiers])
    maxPosition.columns = ['MAXPOSITION']
    data = pd.concat([data, maxPosition], axis=1)
    return data

def average_word_vectors(word_set, word2vec_model):
    """
    Calculate the average word vector for a set of words using a Word2Vec model.

    Args:
        word_set (set): A set of words for which to calculate the average vector.
        word2vec_model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        numpy.ndarray: The average word vector for the input set of words.
    
    Raises:
        ValueError: If none of the words in the set exist in the Word2Vec model.
    """
    word_vectors = []
    for word in word_set:
        if word in word2vec_model.index_to_key:
            word_vectors.append(word2vec_model.get_vector(word))

    if not word_vectors:
        raise ValueError("None of the words in the set exist in the Word2Vec model.")

    return np.mean(word_vectors, axis=0)

def compute_similarity(verb_vector, target_word, model):
    """
    Compute the cosine similarity between a verb vector and a target word vector in a word embedding model.

    Args:
        verb_vector (numpy.ndarray): The vector representation of a verb.
        target_word (str): The target word for which similarity is calculated.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        float: The cosine similarity between the verb vector and the target word vector, or 0.0 if the target word is not in the model.
    """
    try:
        target_word_vector = model.get_vector(key=target_word, norm=True)
        similarity = 1 - cosine(verb_vector, target_word_vector)
        return similarity
    except KeyError:
        return 0.0

def createVerbVectorFeature(data, model):
    """
    Calculate and add a 'VERB_SCORE' column to the DataFrame indicating the similarity of each word to a verb vector.

    This function calculates the average vector of a set of verbs and then computes the cosine similarity between each
    word in the 'WORD' column of the input DataFrame and the verb vector. The similarity scores are added as a new column
    'VERB_SCORE' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'VERB_SCORE' column.
    """
    words = data["WORD"]
    vector = average_word_vectors(verbs, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['VERB_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createDeterminerVectorFeature(data, model):
    """
    Calculate and add a 'DET_SCORE' column to the DataFrame indicating the similarity of each word to a determiner vector.

    This function calculates the average vector of a set of determiners and then computes the cosine similarity between
    each word in the 'WORD' column of the input DataFrame and the determiner vector. The similarity scores are added as
    a new column 'DET_SCORE' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'DET_SCORE' column.
    """
    words = data["WORD"]
    vector = average_word_vectors(conjunctions, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['DET_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createPrepositionVectorFeature(data, model):
    """
    Calculate and add a 'PREP_SCORE' column to the DataFrame indicating the similarity of each word to a preposition vector.

    This function calculates the average vector of a set of prepositions and then computes the cosine similarity between
    each word in the 'WORD' column of the input DataFrame and the preposition vector. The similarity scores are added as
    a new column 'PREP_SCORE' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'PREP_SCORE' column.
    """
    words = data["WORD"]
    vector = average_word_vectors(prepositions, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['PREP_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createConjunctionVectorFeature(data, model):
    """
    Calculate and add a 'CONJ_SCORE' column to the DataFrame indicating the similarity of each word to a conjunction vector.

    This function calculates the average vector of a set of conjunctions and then computes the cosine similarity between
    each word in the 'WORD' column of the input DataFrame and the conjunction vector. The similarity scores are added as
    a new column 'CONJ_SCORE' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'CONJ_SCORE' column.
    """
    words = data["WORD"]
    vector = average_word_vectors(conjunctions, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = ['CONJ_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createPreambleVectorFeature(name, data, model):
    """
    Calculate and add a custom-named preamble similarity score column to the DataFrame.

    This function calculates the similarity between each word in the 'WORD' column of the input DataFrame and a vector
    representation specific to the given 'name' (e.g., 'CODE', 'METHOD', 'ENGLISH'). The similarity scores are added as
    a new column with the provided 'name' and 'PRE_SCORE' appended in the DataFrame.

    Args:
        name (str): The name to use for the custom-named preamble similarity score column.
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional custom-named preamble similarity score column.

    Note:
        The actual name of the new column will be 'name'+'PRE_SCORE' (e.g., 'CODEPRE_SCORE', 'METHODPRE_SCORE').
    """
    words = data["WORD"]
    vector = average_word_vectors(hungarian, model)
    
    scores = pd.DataFrame([compute_similarity(vector, word.lower(), model) for word in words])
    scores.columns = [name+'PRE_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createPrepositionFeature(data):
    """
    Calculate and add a 'PREPOSITION' column to the DataFrame indicating whether each word is a preposition.

    This function checks if each word in the 'WORD' column of the input DataFrame is a preposition and adds a binary
    'PREPOSITION' column (1 for prepositions, 0 otherwise) in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'PREPOSITION' column.
    """
    words = data["WORD"]
    isPreposition = pd.DataFrame([1 if word.lower() in prepositions else 0 for word in words])
    isPreposition.columns = ["PREPOSITION"]
    data = pd.concat([data, isPreposition], axis=1)
    return data

def createConjunctionFeature(data):
    """
    Calculate and add a 'CONJUNCTION' column to the DataFrame indicating whether each word is a conjunction.

    This function checks if each word in the 'WORD' column of the input DataFrame is a conjunction and adds a binary
    'CONJUNCTION' column (1 for conjunctions, 0 otherwise) in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'CONJUNCTION' column.
    """
    words = data["WORD"]
    isConjunction = pd.DataFrame([1 if word.lower() in conjunctions else 0 for word in words])
    isConjunction.columns = ["CONJUNCTION"]
    data = pd.concat([data, isConjunction], axis=1)
    return data


def createDeterminerFeature(data):
    """
    Calculate and add a 'DETERMINER' column to the DataFrame indicating whether each word is a determiner.

    This function checks if each word in the 'WORD' column of the input DataFrame is a determiner and adds a binary
    'DETERMINER' column (1 for determiners, 0 otherwise) in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'DETERMINER' column.
    """
    words = data["WORD"]
    isDeterminer = pd.DataFrame([1 if word.lower() in determiners else 0 for word in words])
    isDeterminer.columns = ["DETERMINER"]
    data = pd.concat([data, isDeterminer], axis=1)
    return data


def createDigitFeature(data):
    """
    Calculate and add a 'DIGITS' column to the DataFrame indicating whether each word consists of digits.

    This function checks if each word in the 'WORD' column of the input DataFrame consists of digits and adds a binary
    'DIGITS' column (1 for words consisting of digits, 0 otherwise) in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'DIGITS' column.
    """
    words = data["WORD"]
    isDigits = pd.DataFrame([1 if word.isdigit() else 0 for word in words])
    isDigits.columns = ["DIGITS"]
    data = pd.concat([data, isDigits], axis=1)
    return data


def createLetterFeature(data):
    """
    Calculate and add a 'LAST_LETTER' column to the DataFrame indicating the ASCII value of the last letter in each word.

    This function calculates the ASCII value of the last letter (converted to lowercase) in each word in the 'WORD' column
    of the input DataFrame and adds this information as a new column 'LAST_LETTER' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'LAST_LETTER' column.
    """
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
    """
    Calculate and add word vector features to the DataFrame.

    This function calculates word vector features for each word in the 'WORD' column of the input DataFrame using the provided
    word embedding model. The word vectors are added as new columns with names 'VEC0', 'VEC1', ... 'VEC(n-1)', where 'n' is
    the dimensionality of the word vectors.

    Args:
        model (Word2Vec): The Word2Vec word embedding model.
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.
        name (str): A name to indicate the type of word vectors (e.g., 'DEFAULT', 'ENG'). Different names may result in
                    different vector dimensions.

    Returns:
        pandas.DataFrame: The input DataFrame with additional word vector feature columns.
    """
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
    """
    Calculate the similarity between two words using a Word2Vec model.

    This function computes the similarity between two words using a Word2Vec model. If both words are present in the model's
    vocabulary, it returns their cosine similarity; otherwise, it returns 0.

    Args:
        word (str): The first word for similarity comparison.
        word2 (str): The second word for similarity comparison.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        float: The cosine similarity between the two words, or 0 if either word is not in the model's vocabulary.
    """
    try:
        # Try to get the word vector from the model
        vector = model.similarity(word, word2)
        return vector
    except KeyError:
        return 0

def createSimilarityToVerbFeature(name, model, data):
    """
    Calculate and add a custom-named similarity score column to the DataFrame indicating the similarity of each word to the word "verb."

    This function calculates the similarity between each word in the 'WORD' column of the input DataFrame and the word "verb"
    using a Word2Vec model. The similarity scores are added as a new column with the provided 'name' and '_SCORE' appended
    in the DataFrame.

    Args:
        name (str): The name to use for the custom-named similarity score column.
        model (Word2Vec): The Word2Vec word embedding model.
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional custom-named similarity score column.

    Note:
        The actual name of the new column will be 'name'+'_SCORE' (e.g., 'METHOD_SCORE', 'ENGLISH_SCORE').
    """
    words = data["WORD"]
    scores = pd.DataFrame([get_word_similarity("verb", word.lower(), model) for word in words])
    scores.columns = [name+'_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createSimilarityToNounFeature(name, model, data):
    """
    Calculate and add a custom-named similarity score column to the DataFrame indicating the similarity of each word to the word "noun."

    This function calculates the similarity between each word in the 'WORD' column of the input DataFrame and the word "noun"
    using a Word2Vec model. The similarity scores are added as a new column with the provided 'name' and '_SCORE' appended
    in the DataFrame.

    Args:
        name (str): The name to use for the custom-named similarity score column.
        model (Word2Vec): The Word2Vec word embedding model.
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional custom-named similarity score column.

    Note:
        The actual name of the new column will be 'name'+'_SCORE' (e.g., 'METHOD_SCORE', 'ENGLISH_SCORE').
    """
    words = data["WORD"]
    scores = pd.DataFrame([get_word_similarity("noun", word.lower(), model) for word in words])
    scores.columns = [name+'_SCORE']
    scores = pd.concat([data, scores], axis=1)
    return scores

def createMethodWordVectorsFeature(model, data):
    """
    Calculate and add method-specific word vector features to the DataFrame.

    This function calculates method-specific word vector features for each word in the 'WORD' column of the input DataFrame
    using the provided word embedding model. The word vectors are added as new columns with names 'MVEC0', 'MVEC1', ... 'MVEC(n-1)',
    where 'n' is the dimensionality of the word vectors.

    Args:
        model (Word2Vec): The Word2Vec word embedding model.
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with additional method-specific word vector feature columns.
    """
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
    """
    Create and load Word2Vec models for tokens, methods, and English text.

    This function loads pre-trained Word2Vec models for tokens, methods, and English text. The models are used for various
    natural language processing tasks.

    Args:
        pklFile (str, optional): The path to a pickle file. Defaults to an empty string.

    Returns:
        tuple: A tuple containing three Word2Vec models: (modelGensimTokens, modelGensimMethods, modelGensimEnglish).
    """
    modelGensimEnglish = api.load('fasttext-wiki-news-subwords-300')
    modelGensimTokens = word2vec.load_word2vec_format('./code2vec/token_vecs.txt', binary=False)
    modelGensimMethods = word2vec.load_word2vec_format('./code2vec/target_vecs.txt', binary=False)

    return modelGensimTokens, modelGensimMethods, modelGensimEnglish