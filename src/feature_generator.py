import time, nltk, sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from typing import List, Callable
from collections import Counter

vector_size = 128

#Conjunctions and determiners are closed set words, so we can soft-code them by doing a lookup on their
#Word embeddings. This avoids the problem with hard-coding (i.e., assuming the word is always a closet set word)
#while still giving our approach the ability to determine if we're in the most-likely context of them being a closed set word
# https://www.talkenglish.com/vocabulary/top-1500-nouns.aspx
nouns = {'people','history','way','art','world','information','map','two','family','government','health','system','computer','meat',
         'year','thanks','music','person','reading','method','data','food','understanding','theory','law','bird','literature','problem',
         'software','control','knowledge','power','ability','economics','love','internet','television','science','library','nature','fact',
         'product','idea','temperature','investment','area','society','activity','story','industry','media','thing','oven','community',
         'definition','safety','quality','development','language','management','player','variety','video','week','security','country',
         'exam','movie','organization','equipment','physics','analysis','policy','series','thought','basis','boyfriend','direction','strategy',
         'technology','army','camera','freedom','paper','environment','child','instance','month','truth','marketing','university','writing',
         'article','department','difference','goal','news','audience','fishing','growth','income','marriage','user','combination','failure',
         'meaning','medicine','philosophy','teacher','communication','night','chemistry','disease','disk','energy','nation','road','role',
         'soup','advertising','location','success','addition','apartment','education','math','moment','painting','politics','attention',
         'decision','event','property','shopping','student','wood','competition','distribution','entertainment','office','population',
         'president','unit','category','cigarette','context','introduction','opportunity','performance','driver','flight','length',
         'magazine','newspaper','relationship','teaching','cell','dealer','finding','lake','member','message','phone','scene','appearance',
         'association','concept','customer','death','discussion','housing','inflation','insurance','mood','woman','advice','blood','effort',
         'expression','importance','opinion','payment','reality','responsibility','situation','skill','statement','wealth','application','city',
         'county','depth','estate','foundation','grandmother','heart','perspective','photo','recipe','studio','topic','collection','depression','imagination',
         'passion','percentage','resource','setting','ad','agency','college','connection','criticism','debt','description','memory','patience','secretary','solution',
         'administration','aspect','attitude','director','personality','psychology','recommendation','response','selection','storage','version','alcohol','argument',
         'complaint','contract','emphasis','highway','loss','membership','possession','preparation','steak','union','agreement','cancer','currency','employment',
         'engineering','entry','interaction','mixture','preference','region','republic','tradition','virus','actor','classroom','delivery','device',
         'difficulty','drama','election','engine','football','guidance','hotel','owner','priority','protection','suggestion','tension','variation',
         'anxiety','atmosphere','awareness','bath','bread','candidate','climate','comparison','confusion','construction','elevator','emotion','employee',
         'employer','guest','height','leadership','mall','manager','operation','recording','sample','transportation','charity','cousin','disaster',
         'editor','efficiency','excitement','extent','feedback','guitar','homework','leader','mom','outcome','permission','presentation','promotion','reflection',
         'refrigerator','resolution','revenue','session','singer','tennis','basket','bonus','cabinet','childhood','church','clothes','coffee','dinner','drawing',
         'hair','hearing','initiative','judgment','lab','measurement','mode','mud','orange','poetry','police','possibility','procedure','queen','ratio','relation',
         'restaurant','satisfaction','sector','signature','significance','song','tooth','town','vehicle','volume','wife','accident','airport','appointment','arrival',
         'assumption','baseball','chapter','committee','conversation','database','enthusiasm','error','explanation','farmer','gate','girl','hall','historian',
         'hospital','injury','instruction','maintenance','manufacturer','meal','perception','pie','poem','presence','proposal','reception','replacement',
         'revolution','river','son','speech','tea','village','warning','winner','worker','writer','assistance','breath','buyer','chest','chocolate','conclusion',
         'contribution','cookie','courage','dad','desk','drawer','establishment','examination','garbage','grocery','honey','impression','improvement',
         'independence','insect','inspection','inspector','king','ladder','menu','penalty','piano','potato','profession','professor','quantity','reaction',
         'requirement','salad','sister','supermarket','tongue','weakness','wedding','affair','ambition','analyst','apple','assignment','assistant','bathroom',
         'bedroom','beer','birthday','celebration','championship','cheek','client','consequence','departure','diamond','dirt','ear','fortune','friendship','funeral',
         'gene','girlfriend','hat','indication','intention','lady','midnight','negotiation','obligation','passenger','pizza','platform','poet','pollution','recognition',
         'reputation','shirt','sir','speaker','stranger','surgery','sympathy','tale','throat','trainer','uncle','youth','time','work','film','water','money','example','while',
         'business','study','game','life','form','air','day','place','number','part','field','fish','back','process','heat','hand','experience','job','book','end','point',
         'type','home','economy','value','body','market','guide','interest','state','radio','course','company','price','size','card','list','mind','trade','line','care','group',
         'risk','word','fat','force','key','light','training','name','school','top','amount','level','order','practice','research','sense','service','piece',
         'web','boss','sport','fun','house','page','term','test','answer','sound','focus','matter','kind','soil','board','oil','picture','access','garden',
         'range','rate','reason','future','site','demand','exercise','image','case','cause','coast','action','age','bad','boat','record','result','section',
         'building','mouse','cash','class','nothing','period','plan','store','tax','side','subject','space','rule','stock','weather','chance','figure','man',
         'model','source','beginning','earth','program','chicken','design','feature','head','material','purpose','question','rock','salt','act','birth','car',
         'dog','object','scale','sun','note','profit','rent','speed','style','war','bank','craft','half','inside','outside','standard','bus','exchange','eye',
         'fire','position','pressure','stress','advantage','benefit','box','frame','issue','step','cycle','face','item','metal','paint','review','room','screen',
         'structure','view','account','ball','discipline','medium','share','balance','bit','black','bottom','choice','gift','impact','machine','shape','tool',
         'wind','address','average','career','culture','morning','pot','sign','table','task','condition','contact','credit','egg','hope','ice','network','north',
         'square','attempt','date','effect','link','post','star','voice','capital','challenge','friend','self','shot','brush','couple','debate','exit','front',
         'function','lack','living','plant','plastic','spot','summer','taste','theme','track','wing','brain','button','click','desire','foot','gas','influence',
         'notice','rain','wall','base','damage','distance','feeling','pair','savings','staff','sugar','target','text','animal','author','budget','discount','file',
         'ground','lesson','minute','officer','phase','reference','register','sky','stage','stick','title','trouble','bowl','bridge','campaign','character','club',
         'edge','evidence','fan','letter','lock','maximum','novel','option','pack','park','plenty','quarter','skin','sort','weight','baby','background','carry','dish',
         'factor','fruit','glass','joint','master','muscle','red','strength','traffic','trip','vegetable','appeal','chart','gear','ideal','kitchen','land','log',
         'mother','net','party','principle','relative','sale','season','signal','spirit','street','tree','wave','belt','bench','commission','copy','drop','minimum',
         'path','progress','project','sea','south','status','stuff','ticket','tour','angle','blue','breakfast','confidence','daughter','degree','doctor','dot','dream',
         'duty','essay','father','fee','finance','hour','juice','limit','luck','milk','mouth','peace','pipe','seat','stable','storm','substance','team','trick',
         'afternoon','bat','beach','blank','catch','chain','consideration','cream','crew','detail','gold','interview','kid','mark','match','mission','pain','pleasure',
         'score','screw','sex','shop','shower','suit','tone','window','agent','band','block','bone','calendar','cap','coat','contest','corner','court','cup',
         'district','door','east','finger','garage','guarantee','hole','hook','implement','layer','lecture','lie','manner','meeting','nose','parking','partner',
         'profile','respect','rice','routine','schedule','swimming','telephone','tip','winter','airline','bag','battle','bed','bill','bother','cake','code','curve',
         'designer','dimension','dress','ease','emergency','evening','extension','farm','fight','gap','grade','holiday','horror','horse','host','husband','loan',
         'mistake','mountain','nail','noise','occasion','package','patient','pause','phrase','proof','race','relief','sand','sentence','shoulder','smoke','stomach',
         'string','tourist','towel','vacation','west','wheel','wine','arm','aside','associate','bet','blow','border','branch','breast','brother','buddy','bunch',
         'chip','coach','cross','document','draft','dust','expert','floor','god','golf','habit','iron','judge','knife','landscape','league','mail','mess','native',
         'opening','parent','pattern','pin','pool','pound','request','salary','shame','shelter','shoe','silver','tackle','tank','trust','assist','bake','bar',
         'bell','bike','blame','boy','brick','chair','closet','clue','collar','comment','conference','devil','diet','fear','fuel','glove','jacket','lunch',
         'monitor','mortgage','nurse','pace','panic','peak','plane','reward','row','sandwich','shock','spite','spray','surprise','till','transition','weekend',
         'welcome','yard','alarm','bend','bicycle','bite','blind','bottle','cable','candle','clerk','cloud','concert','counter','flower','grandfather','harm',
         'knee','lawyer','leather','load','mirror','neck','pension','plate','purple','ruin','ship','skirt','slice','snow','specialist','stroke','switch','trash',
         'tune','zone','anger','award','bid','bitter','boot','bug','camp','candy','carpet','cat','champion','channel','clock','comfort','cow','crack','engineer',
         'entrance','fault','grass','guy','hell','highlight','incident','island','joke','jury','leg','lip','mate','motor','nerve','passage','pen','pride',
         'priest','prize','promise','resident','resort','ring','roof','rope','sail','scheme','script','sock','station','toe','tower','truck','witness'}
#https://7esl.com/conjunctions-list/
conjunctions = {"for", "and", "nor", "but", "or", "yet", "so", "although", "after", "before", "because", "how",
                "if", "once", "since", "until", "unless", "when", "as", "that", "though", "till", "while", "where", "after",
                "although", "as", "lest", "though", "now", "even", "provided", "else", "where", "wherever", "whereas", 
                "whether", "since", "because", "whose", "whoever", "unless", "while", "before", "why", "so that", "until", 
                "how", "since", "than", "till", "whenever", "supposing", "when", "what", "also", "otherwise", "for", "and",  "nor", "but", 
                "so that", "or", "such that", "yet", "as soon as", "so", "also", "whoever", "yet", "while", "still", "until", "too", "unless", 
                "only", "since", "however", "as if", "no less than", "no less than", "which", "otherwise", "where", "in order that", 
                "who", "than", "after", "as", "because", "either or", "whoever", "nevertheless", "though", "else", "although", "if", 
                "while", "till"}
#https://en.wikipedia.org/wiki/List_of_English_determiners
determiners = {"a", "all", "an", "another", "any", "anybody", "anyone", "anything", "anywhere", "both", "certain", "each", 
               "either", "enough", "every", "everybody", "everyone", "everything", "everywhere", "few", "fewer", "fewest", "last", "least", "less", 
               "little", "many", "more", "most", "much", "neither", "next", "no", "no one", "nobody", "none", "nothing", "nowhere", "once", 
               "said", "several", "some", "somebody", "something", "somewhere", "sufficient", "that", "the", "these", "this", "those", "us", 
               "various", "we", "what", "whatever", "which", "whichever", "you"}

#https://en.wikipedia.org/wiki/List_of_English_prepositions
#https://www.englishclub.com/grammar/prepositions-list.php
prepositions = {"aboard", "about", "above", "across", "after", "against", "along", "amid", "among", "anti", "around", "as", "at", "before", "behind", 
                "below", "beneath", "beside", "besides", "between", "beyond", "but", "by", "concerning", "considering", "despite", "down", "during", 
                "except", "excepting", "excluding", "following", "for", "from", "in", "inside", "into", "like", "minus", "near", "of", "off", "on", 
                "onto", "opposite", "outside", "over", "past", "per", "plus", "regarding", "round", "save", "since", "than", "through", "to", "toward", 
                "towards", "under", "underneath", "unlike", "until", "up", "upon", "versus", "via", "with", "within", "without", "out", "till"}

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

def createFeatures(data: pd.DataFrame, feature_list: List[str], modelTokens = None, modelMethods = None, modelGensimEnglish = None) -> pd.DataFrame:
    """
    Create various features for the input data based on the provided feature list.

    Args:
        data (pandas.DataFrame): The input DataFrame containing necessary columns.
        feature_list (List[str]): A list of features to be created.

    Returns:
        pandas.DataFrame: The input DataFrame with additional features added based on the feature list.
    """
    startTime = time.time()

    # Define a mapping of features to their corresponding functions
    feature_function_map: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        'NORMALIZED_POSITION': createNormalizedPositionFeature,
        'NLTK_POS': wordPosTag,
        'POSITION': createIdentifierPositionFeature,
        'MAXPOSITION': createIdentifierMaxPositionFeature,
        'POSITION_RATIO': positionRatio,
        'METHOD_VERB_SCORE': lambda df: createVerbVectorFeature("METHOD", df, modelMethods),
        'TOKEN_VERB_SCORE': lambda df: createVerbVectorFeature("TOKEN", df, modelTokens),
        'ENGLISH_VERB_SCORE': lambda df: createVerbVectorFeature("ENGLISH", df, modelGensimEnglish),
        'METHOD_NOUN_SCORE': lambda df: createNounVectorFeature("METHOD", df, modelGensimEnglish),
        'TOKEN_NOUN_SCORE': lambda df: createNounVectorFeature("TOKEN", df, modelGensimEnglish),
        'ENGLISH_NOUN_SCORE': lambda df: createNounVectorFeature("ENGLISH", df, modelGensimEnglish),
        'EMB_FEATURES': lambda df: createEmbeddingFeatures(df, modelGensimEnglish),
        'DET_SCORE': lambda df: createDeterminerVectorFeature(df, modelGensimEnglish),
        'PREP_SCORE': lambda df: createPrepositionVectorFeature(df, modelGensimEnglish),
        'CONJ_SCORE': lambda df: createConjunctionVectorFeature(df, modelGensimEnglish),
        'CODEPRE_SCORE': lambda df: createPreambleVectorFeature("CODE", df, modelTokens),
        'METHODPRE_SCORE': lambda df: createPreambleVectorFeature("METHOD", df, modelMethods),
        'ENGLISHPRE_SCORE': lambda df: createPreambleVectorFeature("ENGLISH", df, modelGensimEnglish),
        'CONTAINSLISTVERB': createVerbFeature,
        'PREPOSITION': createPrepositionFeature,
        'CONJUNCTION': createConjunctionFeature,
        'DETERMINER': createDeterminerFeature,
        'DIGITS': createDigitFeature,
        'CONTAINSDIGITS': createIdentifierDigitFeature,
        'CONTAINSCLOSEDSET': createIdentifierClosedSetFeature,
        'CONTAINSVERB': createIdentifierContainsVerbFeature,
        'LAST_LETTER': createLetterFeature,
        'MORPHOLOGICAL_PLURAL': addMorphologicalPluralFeature,
        'CONSONANT_VOWEL_RATIO': consonantVowelRatio,
        'SECOND_LAST_LETTER': createLastTwoLettersFeature,
        'METHODV_SCORE': lambda df: createSimilarityToVerbFeature("METHODV", modelMethods, df),
        'ENGLISHV_SCORE': lambda df: createSimilarityToVerbFeature("ENGLISHV", modelGensimEnglish, df),
        'METHODN_SCORE': lambda df: createSimilarityToNounFeature("METHODN", modelMethods, df),
        'ENGLISHN_SCORE': lambda df: createSimilarityToNounFeature("ENGLISHN", modelGensimEnglish, df),
    }

    # Apply functions based on the feature list
    for feature in feature_list:
        if feature in feature_function_map:
            data = feature_function_map[feature](data)
        else:
            print(f"Warning: Feature '{feature}' not found in the feature function map.")

    print(f"Total Feature Time: {time.time() - startTime}")
    return data

universal_to_custom = {
    # Nouns
    'NN': 'NOUN',     # Noun, singular or mass
    'NNS': 'NOUN_PL', # Noun, plural (special addition for plural nouns)
    'NNP': 'NOUN',    # Proper noun, singular
    'NNPS': 'NOUN_PL', # Proper noun, plural

    # Verbs
    'VB': 'VERB',     # Verb, base form
    'VBD': 'VERB',    # Verb, past tense
    'VBG': 'VERB',    # Verb, gerund or present participle
    'VBN': 'VERB',    # Verb, past participle
    'VBP': 'VERB',    # Verb, non-3rd person singular present
    'VBZ': 'VERB',    # Verb, 3rd person singular present
    'MD': 'VERB',     # Modal

    # Adjectives
    'JJ': 'ADJ',      # Adjective
    'JJR': 'ADJ',     # Adjective, comparative
    'JJS': 'ADJ',     # Adjective, superlative

    # Adverbs
    'RB': 'ADV',      # Adverb
    'RBR': 'ADV',     # Adverb, comparative
    'RBS': 'ADV',     # Adverb, superlative
    'WRB': 'ADV',     # Wh-adverb

    # Determiners and Pronouns
    'DT': 'DET',      # Determiner
    'WDT': 'DET',     # Wh-determiner
    'PDT': 'DET',     # Predeterminer
    'PRP': 'DET',     # Personal pronoun
    'PRP$': 'DET',    # Possessive pronoun
    'WP': 'DET',      # Wh-pronoun
    'WP$': 'DET',     # Possessive wh-pronoun
    'EX': 'DET',      # Existential there

    # Prepositions and Conjunctions
    'IN': 'ADP',      # Preposition or subordinating conjunction
    'TO': 'ADP',      # to
    'CC': 'CONJ',     # Coordinating conjunction

    # Numbers and Symbols
    'CD': 'NUM',      # Cardinal number
    'LS': 'NUM',      # List item marker
    'SYM': 'NOUN',    # Symbol

    # Others
    'UH': 'NOUN',     # Interjection
    'FW': 'NOUN',     # Foreign word
    'POS': 'NOUN',    # Possessive ending
    'RP': 'NOUN',     # Particle
    'X': 'NOUN',      # Unknown

    'START':'START',
    'END':'END',

    # Punctuation
    '.': '.',         # Punctuation
}

custom_to_numeric = {
    'VERB': 0,
    'NOUN': 1,
    'ADJ': 2,
    'ADV': 3,
    'ADP': 4,
    'CONJ': 5,
    'DET': 6,
    'NUM': 7,
    'NOUN_PL': 8,
    '.': 9,
    'START':10,
    'END':11
}

def wordPosTag(data):
    """
    Perform part-of-speech tagging with context-aware tagging for entire identifiers before assigning word-level POS.

    Args:
        data (pandas.DataFrame): Input DataFrame with 'WORD' and 'SPLIT_IDENTIFIER' columns.

    Returns:
        pandas.DataFrame: DataFrame with NLTK_POS, PREV_POS, NEXT_POS columns.
    """
    # Validate columns
    if 'WORD' not in data.columns or 'SPLIT_IDENTIFIER' not in data.columns:
        raise ValueError("DataFrame must contain 'WORD' and 'SPLIT_IDENTIFIER' columns")
    
    # Function to tag words within each identifier
    def process_identifier_group(group):
        identifier = " ".join(group['WORD'])  # Reconstruct full identifier for better context
        tokens = nltk.word_tokenize(identifier)  # Tokenize the identifier
        pos_tags = nltk.pos_tag(tokens)  # Get POS tags
        
        # Create mapping of tokens to POS tags
        pos_dict = {word: tag for word, tag in pos_tags}

        # Assign POS tags to words in the group
        group['NLTK_POS'] = group['WORD'].map(lambda word: pos_dict.get(word, 'NN'))  # Default to NN (noun)
        
        # Assign previous and next POS tags
        pos_list = group['NLTK_POS'].tolist()
        group['PREV_POS'] = ['START'] + pos_list[:-1]
        group['NEXT_POS'] = pos_list[1:] + ['END']
        
        return group

    # Apply grouping by identifier and process each identifier
    result = data.groupby('SPLIT_IDENTIFIER').apply(process_identifier_group).reset_index(drop=True)
    
    return result

def calculate_consonant_vowel_ratio(word):
    """
    Calculate the ratio of consonants to vowels in a word.
    
    Args:
        word (str): The input word to analyze.
    
    Returns:
        float: Ratio of consonants to vowels, defaulting to 0 in edge cases.
    """
    # Ensure input is a string and lowercase
    word = str(word).lower()
    
    vowels = set('aeiou')
    # Filter for alphabetic consonants
    consonants = [c for c in word if c.isalpha() and c not in vowels]
    vowel_count = sum(1 for c in word if c in vowels)
    
    # Return 0 if no alphabetic characters or no vowels
    if not word or vowel_count == 0:
        return 0.0
    
    return len(consonants) / max(vowel_count, 1)

def createIdentifierPositionFeature(data):
    """
    Add a 'POSITION' column indicating the position of the current word in the identifier.
    """
    # Initialize the POSITION column
    positions = []

    # Iterate through the rows
    for _, row in data.iterrows():
        words = row['SPLIT_IDENTIFIER'].lower().split()
        current_word = row['WORD']
        
        # Find the position of the current word in the identifier (1-based index)
        try:
            position = words.index(current_word) + 1
        except ValueError:
            position = 0  # If the word is not found (fallback case)
        
        positions.append(position)
    
    # Assign the calculated positions
    data['POSITION'] = positions
    return data

def consonantVowelRatio(data):
    """
    Add a CONSONANT_VOWEL_RATIO feature to the DataFrame.
    Args:
        data (pd.DataFrame): Input DataFrame with 'WORD' column.
    Returns:
        pd.DataFrame: DataFrame with added CONSONANT_VOWEL_RATIO column.
    """
    consonant_vowel_ratios = data["WORD"].apply(calculate_consonant_vowel_ratio)
    data["CONSONANT_VOWEL_RATIO"] = consonant_vowel_ratios.fillna(0.0)
    return data

def createIdentifierMaxPositionFeature(data):
    """
    Add a 'MAXPOSITION' column indicating the total number of words in each identifier.
    """
    # Calculate max position for each unique identifier
    data['MAXPOSITION'] = data['SPLIT_IDENTIFIER'].str.split().str.len()
    return data

def positionRatio(data):
    data["POSITION_RATIO"] = data["POSITION"].astype(int) / data["MAXPOSITION"].replace(0, pd.NA)
    data["POSITION_RATIO"] = data["POSITION_RATIO"].fillna(0)
    return data
    
def average_word_vectors(word_set, word2vec_model):
    """
    Calculate the average word vector for a set of words using a Word2Vec model, with normalization.

    Args:
        word_set (set): A set of words for which to calculate the average vector.
        word2vec_model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        numpy.ndarray: The normalized average word vector for the input set of words.
    
    Raises:
        ValueError: If none of the words in the set exist in the Word2Vec model.
    """
    vectors = [word2vec_model[word] for word in word_set if word in word2vec_model.key_to_index]
    
    if not vectors:
        raise ValueError("None of the words in the set exist in the Word2Vec model.")
    
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector / np.linalg.norm(avg_vector)  # Normalize the result

def compute_similarity(verb_vector, target_word, model):
    """
    Compute the cosine similarity between a normalized verb vector and a normalized target word vector in a word embedding model.

    Args:
        verb_vector (numpy.ndarray): The vector representation of a verb (will be normalized within the function).
        target_word (str): The target word for which similarity is calculated.
        model (Word2Vec): The Word2Vec word embedding model.

    Returns:
        float: The cosine similarity between the normalized verb vector and the normalized target word vector,
               or 0.0 if the target word is not in the model.
    """
    if target_word not in model.key_to_index:
        return 0.0

    # Normalize the verb vector
    verb_vector = verb_vector / np.linalg.norm(verb_vector)

    # Get and normalize the target word vector
    target_word_vector = model[target_word]
    target_word_vector = target_word_vector / np.linalg.norm(target_word_vector)

    # Compute cosine similarity
    similarity = np.dot(verb_vector, target_word_vector)
    return similarity


def contrastive_embedding(target_vector, contrast_vectors, beta=0.1):
    """
    Adjust a target embedding by pushing it away from contrast embeddings.

    Args:
        target_vector (np.array): Base vector (e.g., verb or noun vector).
        contrast_vectors (list of np.array): List of contrast word embeddings.
        beta (float): Strength of repulsion from contrast embeddings.

    Returns:
        np.array: Adjusted embedding.
    """
    if not contrast_vectors:
        return target_vector  # No contrast data, return original

    contrast_mean = np.mean(contrast_vectors, axis=0)
    return target_vector + beta * (target_vector - contrast_mean)  # Push away from contrast mean


def createVerbVectorFeature(ID, data, model):
    """
    Compute 'VERB_SCORE' by measuring word similarity to a contrast-enhanced verb vector.
    The embedding is anchored with the word "Verb" to reinforce its representation.

    Args:
        ID (str): Identifier for column naming.
        data (pandas.DataFrame): DataFrame with a 'WORD' column.
        model (Word2Vec): Word2Vec embedding model.

    Returns:
        pandas.DataFrame: Updated DataFrame with a 'VERB_SCORE' column.
    """
    words = data["WORD"]
    
    # Convert sets to lists before merging
    non_verbs = list(nouns) + list(conjunctions) + list(determiners) + list(prepositions)
    
    # Include "Verb" as an anchor in the verb set
    expanded_verbs = list(verbs) + ["verb"] * 5  # Adding weight by repetition
    
    # Compute base verb and non-verb vectors
    verb_vector = average_word_vectors(expanded_verbs, model)
    non_verb_vector = average_word_vectors(non_verbs, model)
    
    # Apply contrastive shift
    adjusted_verb_vector = contrastive_embedding(verb_vector, [non_verb_vector], beta=0.5)
    
    # Compute similarity
    scores = pd.DataFrame([compute_similarity(adjusted_verb_vector, word.lower(), model) for word in words])
    scores.columns = [ID+'_VERB_SCORE']
    
    return pd.concat([data, scores], axis=1)


def createNounVectorFeature(ID, data, model):
    """
    Compute 'NOUN_SCORE' by measuring word similarity to a contrast-enhanced noun vector.

    Args:
        ID (str): Identifier for column naming.
        data (pandas.DataFrame): DataFrame with a 'WORD' column.
        model (Word2Vec): Word2Vec embedding model.

    Returns:
        pandas.DataFrame: Updated DataFrame with a 'NOUN_SCORE' column.
    """
    words = data["WORD"]
    
    # Convert sets to lists before merging
    verb_list = list(verbs) + list(conjunctions) + list(determiners) + list(prepositions)
    noun_list = list(nouns) + ["noun"] * 5
    
    # Compute base noun and verb vectors
    noun_vector = average_word_vectors(noun_list, model)
    verb_vector = average_word_vectors(verb_list, model)
    
    # Apply contrastive shift
    adjusted_noun_vector = contrastive_embedding(noun_vector, [verb_vector], beta=0.5)
    
    # Compute similarity
    scores = pd.DataFrame([compute_similarity(adjusted_noun_vector, word.lower(), model) for word in words])
    scores.columns = [ID+'_NOUN_SCORE']
    
    return pd.concat([data, scores], axis=1)

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
    
    # Convert sets to lists before merging
    non_determiners = list(nouns) + list(verbs) + list(prepositions) +  list(conjunctions)
    
    # Compute base verb and non-verb vectors
    expanded_determiners = list(determiners) + ["determiner"] * 5  # Adding weight by repetition
    determiner_vector = average_word_vectors(expanded_determiners, model)
    non_determiner_vector = average_word_vectors(non_determiners, model)
    
    # Apply contrastive shift
    adjusted_determiner_vector = contrastive_embedding(determiner_vector, [non_determiner_vector], beta=0.1)
    
    # Compute similarity
    scores = pd.DataFrame([compute_similarity(adjusted_determiner_vector, word.lower(), model) for word in words])

    scores.columns = ['DET_SCORE']

    data = pd.concat([data, scores], axis=1)

    return data

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

    # Convert sets to lists before merging
    non_prepositions = list(nouns) + list(verbs) + list(determiners) + list(conjunctions)
    
    # Compute base verb and non-verb vectors
    expanded_prepositions = list(prepositions) + ["preposition"] * 5  # Adding weight by repetition
    preposition_vector = average_word_vectors(expanded_prepositions, model)
    non_preposition_vector = average_word_vectors(non_prepositions, model)
    
    # Apply contrastive shift
    adjusted_preposition_vector = contrastive_embedding(preposition_vector, [non_preposition_vector], beta=0.1)
    
    # Compute similarity
    scores = pd.DataFrame([compute_similarity(adjusted_preposition_vector, word.lower(), model) for word in words])
    scores.columns = ['PREP_SCORE']
    return pd.concat([data, scores], axis=1)

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
    
    # Convert sets to lists before merging
    non_conjunctions = list(nouns) + list(verbs) + list(determiners) + list(prepositions)
    
    # Compute base verb and non-verb vectors
    expanded_conjunctions = list(conjunctions) + ["conjunction"] * 5  # Adding weight by repetition
    conjunction_vector = average_word_vectors(expanded_conjunctions, model)
    non_conjunction_vector = average_word_vectors(non_conjunctions, model)
    
    # Apply contrastive shift
    adjusted_conjunction_vector = contrastive_embedding(conjunction_vector, [non_conjunction_vector], beta=0.1)
    
    # Compute similarity
    scores = pd.DataFrame([compute_similarity(adjusted_conjunction_vector, word.lower(), model) for word in words])
    scores.columns = ['CONJ_SCORE']
    
    return pd.concat([data, scores], axis=1)

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

    # Convert sets to lists before merging
    non_preambles = list(nouns) + list(verbs) + list(determiners) + list(prepositions) + list(determiners)
    
    # Compute base verb and non-verb vectors
    expanded_preambles = list(hungarian) + ["hungarian notation"] * 5  # Adding weight by repetition
    preamble_vector = average_word_vectors(expanded_preambles, model)
    non_preamble_vector = average_word_vectors(non_preambles, model)
    
    # Apply contrastive shift
    adjusted_preamble_vector = contrastive_embedding(preamble_vector, [non_preamble_vector], beta=0.1)
    
    # Compute similarity
    scores = pd.DataFrame([compute_similarity(adjusted_preamble_vector, word.lower(), model) for word in words])
    scores.columns = [name+'PRE_SCORE']
    
    return pd.concat([data, scores], axis=1)

def createVerbFeature(data):
    """
    Calculate and add a 'VERB' column to the DataFrame indicating whether each word is a preposition.

    This function checks if each word in the 'WORD' column of the input DataFrame is a preposition and adds a binary
    'VERB' column (1 for verbs, 0 otherwise) in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'VERB' column.
    """
    words = data["WORD"]
    isVerb = pd.DataFrame([1 if word.lower() in verbs else 0 for word in words])
    isVerb.columns = ["CONTAINSLISTVERB"]
    data = pd.concat([data, isVerb], axis=1)
    return data

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

def createNormalizedPositionFeature(data):
    """
    Add a 'NORMALIZED_POSITION' column indicating the normalized position:
    0 for the first word, 1 for middle words, and 2 for the final word.
    """
    # Initialize the NORMALIZED_POSITION column
    normalized_positions = []

    # Iterate through the rows
    for _, row in data.iterrows():
        words = row['SPLIT_IDENTIFIER'].lower().split()
        current_word = row['WORD']
        
        # Determine the normalized position
        if current_word == words[0]:
            normalized_position = 0  # First word
        elif current_word == words[-1]:
            normalized_position = 2  # Final word
        else:
            normalized_position = 1  # Middle words
        
        normalized_positions.append(normalized_position)
    
    # Assign the calculated normalized positions
    data['NORMALIZED_POSITION'] = normalized_positions
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

def createIdentifierDigitFeature(data):
    """
    Add a 'CONTAINSDIGITS' column indicating if any word in the SPLIT_IDENTIFIER contains digits.

    Args:
        data (pandas.DataFrame): Input DataFrame with 'SPLIT_IDENTIFIER' as a list of words.

    Returns:
        pandas.DataFrame: Updated DataFrame with a 'CONTAINSDIGITS' column.
    """
    # Check if any word in SPLIT_IDENTIFIER is a digit
    data['CONTAINSDIGITS'] = data['SPLIT_IDENTIFIER'].apply(
        lambda words: any(word.isdigit() for word in words)
    ).astype(int)
    return data

def createEmbeddingFeatures(data, model):
    """
    Add statistical features (mean, std, min, max, norm) from FastText/Word2Vec embeddings.

    Args:
        data (pandas.DataFrame): Input DataFrame with 'SPLIT_IDENTIFIER' as a list of words.

    Returns:
        pandas.DataFrame: Updated DataFrame with new embedding-based numerical features.
    """
    def compute_embedding_stats(words):
        # Get embeddings for words that exist in the model
        word_vectors = [model[word] for word in words if word in model]
        
        if not word_vectors:  # Handle cases where no words have embeddings
            return np.zeros(5)  # Return a zero vector with 5 statistical features

        word_vectors = np.array(word_vectors)
        
        # Compute statistical features
        mean_embedding = np.mean(word_vectors)
        std_embedding = np.std(word_vectors)
        min_embedding = np.min(word_vectors)
        max_embedding = np.max(word_vectors)
        norm_embedding = np.linalg.norm(np.mean(word_vectors, axis=0))

        return np.array([mean_embedding, std_embedding, min_embedding, max_embedding, norm_embedding])

    # Apply function to each row and expand into separate columns
    embedding_features = data['SPLIT_IDENTIFIER'].apply(compute_embedding_stats).apply(pd.Series)
    embedding_features.columns = ['EMB_MEAN', 'EMB_STD', 'EMB_MIN', 'EMB_MAX', 'EMB_NORM']

    return pd.concat([data, embedding_features], axis=1)

def createIdentifierClosedSetFeature(data, conjunctions=conjunctions, determiners=determiners, prepositions=prepositions):
    """
    Add a 'CONTAINSCLOSEDSET' column indicating if any word in the SPLIT_IDENTIFIER matches closed-set words.

    Args:
        data (pandas.DataFrame): Input DataFrame with 'SPLIT_IDENTIFIER' as a list of words.
        conjunctions (set): Set of conjunction words.
        determiners (set): Set of determiner words.
        prepositions (set): Set of preposition words.

    Returns:
        pandas.DataFrame: Updated DataFrame with a 'CONTAINSCLOSEDSET' column.
    """
    closed_set = set(conjunctions) | set(determiners) | set(prepositions)

    words = data["WORD"]
    isClosedSet = pd.DataFrame([1 if word in closed_set else 0 for word in words])
    isClosedSet.columns = ["CONTAINSCLOSEDSET"]
    data = pd.concat([data, isClosedSet], axis=1)
    
    return data

def createIdentifierContainsVerbFeature(data, verbs=verbs):
    """
    Add a 'CONTAINSVERB' column indicating if any word in the SPLIT_IDENTIFIER is a verb.

    Args:
        data (pandas.DataFrame): Input DataFrame with 'SPLIT_IDENTIFIER' as a list of words.
        verbs (set): Set of verb words.

    Returns:
        pandas.DataFrame: Updated DataFrame with a 'CONTAINSVERB' column.
    """
    verb_set = set(verbs)
    
    words = data["WORD"]
    isVerb = pd.DataFrame([1 if word in verb_set else 0 for word in words])
    isVerb.columns = ["CONTAINSVERB"]
    data = pd.concat([data, isVerb], axis=1)
    
    return data

def addMorphologicalPluralFeature(data):
    """
    Add a 'MORPHOLOGICAL_PLURAL' column to indicate morphological plural patterns.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'MORPHOLOGICAL_PLURAL' column.
    """
    def detect_morphological_plural(word):
        if not isinstance(word, str):
            return False
        word = word.lower()
        if word.endswith('ies') and len(word) > 3:  # Example: cities
            return True
        if word.endswith('ves') and len(word) > 3:  # Example: wolves
            return True
        if word.endswith('es') and not word.endswith('ses'):  # Avoid cases like "glasses"
            return True
        if word.endswith('s') and len(word) > 1:  # General plural case
            return True
        return False
    
    data['MORPHOLOGICAL_PLURAL'] = data['WORD'].apply(detect_morphological_plural)
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

def createLastTwoLettersFeature(data):
    """
    Calculate and add a 'SECOND_LAST_LETTER' column to the DataFrame indicating the concatenated ASCII values of the last two letters in each word.

    This function calculates the ASCII values of the last two letters (converted to lowercase) in each word in the 'WORD' column
    of the input DataFrame and adds this information as a new column 'SECOND_LAST_LETTER' in the DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a 'WORD' column.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'SECOND_LAST_LETTER' column.
    """
    lastTwoLetters = pd.DataFrame(np.array([ord(word[-2:].lower()[0]) if len(word) > 1 else ord(word.lower()) for word in data["WORD"]]))
    lastTwoLetters.columns = ["SECOND_LAST_LETTER"]
    data = pd.concat([data, lastTwoLetters], axis=1)
    return data

def get_word_vector(word, model, vector_size):
    try:
        # Try to get the word vector from the model
        vector = model.get_vector(word)
        return vector
    except KeyError:
        return np.zeros(model.vector_size)            

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