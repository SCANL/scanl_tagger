# SCALAR Part-of-speech tagger
This the official release of the SCALAR Part-of-speech tagger

There are two ways to run the tagger. This document describes both ways.

1. Using Docker compose (which runs the tagger's built-in server for you)
2. Running the tagger's built-in server without Docker

## Getting Started with Docker

To run SCNL tagger in a Docker container you can clone the repository and pull the latest docker impage from `sourceslicer/scalar_tagger:latest`

Make sure you have Docker and Docker Compose installed:
https://docs.docker.com/engine/install/
https://docs.docker.com/compose/install/

```
git clone git@github.com:SCANL/scanl_tagger.git
cd scanl_tagger
docker compose pull
docker compose up
```

## Getting Started without Docker
You will need `python3.12` installed. 

You'll need to install `pip` -- https://pip.pypa.io/en/stable/installation/

Set up a virtual environtment: `python -m venv /tmp/tagger` -- feel free to put it somewhere else (change /tmp/tagger) if you prefer

Activate the virtual environment: `source /tmp/tagger/bin/activate` (you can find how to activate it here if `source` does not work for you -- https://docs.python.org/3/library/venv.html#how-venvs-work)

After it's installed and your virtual environment is activated, in the root of the repo, run `pip install -r requirements.txt`

Finally, we require the `token` and `target` vectors from [code2vec](https://github.com/tech-srl/code2vec). The tagger will attempt to automatically download them if it doesn't find them, but you could download them yourself if you like. It will place them in your local directory under `./code2vec/*`

## Usage

```
usage: main [-h] [-v] [-r] [-t] [-a ADDRESS] [--port PORT] [--protocol PROTOCOL]
            [--words WORDS]

options:
  -h, --help            show this help message and exit
  -v, --version         print tagger application version
  -r, --run             run server for part of speech tagging requests
  -t, --train           run training set to retrain the model
  -a ADDRESS, --address ADDRESS
                        configure server address
  --port PORT           configure server port
  --protocol PROTOCOL   configure whether the server uses http or https
  --words WORDS         provide path to a list of acceptable abbreviations
```

`./main -r` will start the server, which will listen for identifier names sent via HTTP over the route:

http://127.0.0.1:8080/{cache_selection}/{identifier_name}/{code_context}

"cache selection" will save results to a separate cache if it is set to "student"

"code context" is one of:
- FUNCTION
- ATTRIBUTE
- CLASS
- DECLARATION
- PARAMETER

For example:

Tag a declaration: ``http://127.0.0.1:8080/cache/numberArray/DECLARATION``

Tag a function: ``http://127.0.0.1:8080/cache/GetNumberArray/FUNCTION``

Tag an class: ``http://127.0.0.1:8080/cache/PersonRecord/CLASS``

#### Note
Kebab case is not currently supported due to the limitations of Spiral. Attempting to send the tagger identifiers which are in kebab case will result in the entry of a single noun. 

You will need to have a way to parse code and filter out identifier names if you want to do some on-the-fly analysis of source code. We recommend [srcML](https://www.srcml.org/). Since the actual tagger is a web server, you don't have to use srcML. You could always use other AST-based code representations, or any other method of obtaining identifier information. 


## Tagset

**Supported Tagset**
| Abbreviation |                 Expanded Form                |                   Examples                   |
|:------------:|:--------------------------------------------:|:--------------------------------------------:|
|       N      |                     noun                     | Disneyland, shoe, faucet, mother             |
|      DT      |                  determiner                  | the, this, that, these, those, which         |
|      CJ      |                  conjunction                 | and, for, nor, but, or, yet, so              |
|       P      |                  preposition                 | behind, in front of, at, under, above        |
|      NPL     |                  noun plural                 | Streets, cities, cars, people, lists         |
|      NM      | noun modifier  (**noun-adjunct**, adjective) | red, cold, hot, **bit**Set, **employee**Name |
|       V      |                     verb                     | Run, jump, spin,                             |
|      VM      |            verb modifier  (adverb)           | Very, loudly, seriously, impatiently         |
|       D      |                     digit                    | 1, 2, 10, 4.12, 0xAF                         |
|      PRE     |                   preamble                   | Gimp, GLEW, GL, G, p, m, b                   |

**Penn Treebank to SCALAR tagset**

|   Penn Treebank Annotation  | SCALAR Tagset            |
|:---------------------------:|:------------------------:|
|       Conjunction (CC)      |     Conjunction (CJ)     |
|          Digit (CD)         |         Digit (D)        |
|       Determiner (DT)       |      Determiner (DT)     |
|      Foreign Word (FW)      |         Noun (N)         |
|       Preposition (IN)      |      Preposition (P)     |
|        Adjective (JJ)       |    Noun Modifier (NM)    |
| Comparative Adjective (JJR) |    Noun Modifier (NM)    |
| Superlative Adjective (JJS) |    Noun Modifier (NM)    |
|        List Item (LS)       |         Noun (N)         |
|          Modal (MD)         |         Verb (V)         |
|      Noun Singular (NN)     |         Noun (N)         |
|      Proper Noun (NNP)      |         Noun (N)         |
|  Proper Noun Plural (NNPS)  |     Noun Plural (NPL)    |
|      Noun Plural (NNS)      |     Noun Plural (NPL)    |
|         Adverb (RB)         |    Verb Modifier (VM)    |
|   Comparative Adverb (RBR)  |    Verb Modifier (VM)    |
|        Particle (RP)        |    Verb Modifier (VM)    |
|         Symbol (SYM)        |         Noun (N)         |
|     To Preposition (TO)     |      Preposition (P)     |
|          Verb (VB)          |         Verb (V)         |
|          Verb (VBD)         |         Verb (V)         |
|          Verb (VBG)         |         Verb (V)         |
|          Verb (VBN)         |         Verb (V)         |
|          Verb (VBP)         |         Verb (V)         |
|          Verb (VBZ)         |         Verb (V)         |

## Training the tagger
You can train this tagger using the `-t` option (which will re-run the training routine). For the moment, most of this is hard-coded in, so if you want to use a different data set/different seeds, you'll need to modify the code. This will potentially change in the future.

## Errors?
Please make an issue if you run into errors

# Please Cite the Paper(s)!

Newman, Christian, Scholten , Brandon, Testa, Sophia, Behler, Joshua, Banabilah, Syreen, Collard, Michael L., Decker, Michael, Mkaouer, Mohamed Wiem, Zampieri, Marcos, Alomar, Eman Abdullah, Alsuhaibani, Reem, Peruma, Anthony, Maletic, Jonathan I., (2025), “SCALAR: A Part-of-speech Tagger for Identifiers”, in the Proceedings of the 33rd IEEE/ACM International Conference on Program Comprehension - Tool Demonstrations Track (ICPC), Ottawa, ON, Canada, April 27 -28, 5 pages TO APPEAR.

Christian  D.  Newman,  Michael  J.  Decker,  Reem  S.  AlSuhaibani,  Anthony  Peruma,  Satyajit  Mohapatra,  Tejal  Vishnoi, Marcos Zampieri, Mohamed W. Mkaouer, Timothy J. Sheldon, and Emily Hill, "An Ensemble Approach for Annotating Source Code Identifiers with Part-of-speech Tags," in IEEE Transactions on Software Engineering, doi: 10.1109/TSE.2021.3098242.

# Training set
The data used to train this tagger can be found in the most recent database update in the repo -- https://github.com/SCANL/scanl_tagger/blob/master/input/scanl_tagger_training_db_11_29_2024.db

# Interested in our other work?
Find our other research [at our webpage](https://www.scanl.org/) and check out the [Identifier Name Structure Catalogue](https://github.com/SCANL/identifier_name_structure_catalogue)

# WordNet
This project uses WordNet to perform a dictionary lookup on the individual words in each identifier:

Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010

