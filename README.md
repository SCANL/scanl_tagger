# SCALAR Part-of-speech tagger
This the official release of the SCALAR Part-of-speech tagger

**NOTE**
There is a fork of SCALAR which was designed to handle parallel http requests and cache SCALAR's output to increase its speed. You can find this version here: https://github.com/brandonscholten/scanl_tagger. These will be combined into a single application in the *very* near future.

## Setup and Run
You will need `python3` installed. We will explicitly use the `python3` command below but, of course, if your environment is configured to use python3 by default, you do not need to. We have also only tested this on **Ubuntu 22** and **Ubuntu via WSL**. It most likely works in similar environments, but no guarantees.

You'll need to install `pip3`

Conosider configuring `PYTHONPATH` as well:

	export PYTHONPATH=~/path/to/scanl_tagger

Finally, you need to install Spiral, which we use for identifier splitting. The current version of Spiral on the official repo has a [problem](https://github.com/casics/spiral/issues/4), so consider installing the one from the link below:

    sudo pip3 install git+https://github.com/cnewman/spiral.git

Finally, we require the `token` and `target` vectors from [code2vec](https://github.com/tech-srl/code2vec). The tagger will attempt to automatically download them if it doesn't find them, but you could download them yourself if you like. It will place them in your local directory under `./code2vec/*`

## Usage

```bash
python main.py -v  # Display the application version.
python main.py -r  # Start the server for tagging requests.
python main.py -t  # Run the training set to retrain the model.
```

`python main.py -r` will start the server, which will listen for identifier names sent via HTTP over the route:

http://127.0.0.1:5000/{identifier_name}/{code_context}

Where "code context" is one of:
- FUNCTION
- ATTRIBUTE
- CLASS
- DECLARATION
- PARAMETER

For example:

Tag a declaration: ``http://127.0.0.1:5000/numberArray/DECLARATION``

Tag a function: ``http://127.0.0.1:5000/GetNumberArray/FUNCTION``

Tag an class: ``http://127.0.0.1:5000/PersonRecord/CLASS``

You will need to have a way to parse code and filter out identifier names if you want to do some on-the-fly analysis of source code. We recommend [srcML](https://www.srcml.org/). Since the actual tagger is a web server, you don't have to use srcML. You could always use other AST-based code representations, or any other method of obtaining identifier information. 

## Training the tagger
You can train this tagger using the `-t` option (which will re-run the training routine). For the moment, most of this is hard-coded in, so if you want to use a different data set/different seeds, you'll need to modify the code. This is will potentially change in the future.

## Errors?
Please make an issue if you run into errors

# Please Cite the Paper!

No paper for now however the current tagger is based on our previous, so you could cite the previous one for now: 

Christian  D.  Newman,  Michael  J.  Decker,  Reem  S.  AlSuhaibani,  Anthony  Peruma,  Satyajit  Mohapatra,  Tejal  Vishnoi, Marcos Zampieri, Mohamed W. Mkaouer, Timothy J. Sheldon, and Emily Hill, "An Ensemble Approach for Annotating Source Code Identifiers with Part-of-speech Tags," in IEEE Transactions on Software Engineering, doi: 10.1109/TSE.2021.3098242.

# Training set
The data used to train this tagger can be found in the most recent database update in the repo -- https://github.com/SCANL/scanl_tagger/blob/master/input/scanl_tagger_training_db_11_29_2024.db

# Interested in our other work?
Find our other research [at our webpage](https://www.scanl.org/) and check out the [Identifier Name Structure Catalogue](https://github.com/SCANL/identifier_name_structure_catalogue)
