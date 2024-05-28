# SCANL tagger 
This the official release of the SCANL part-of-speech tagger.

## Setup and Run
You will need `python3.10` installed. 

You'll need to install `pip3`

Conosider configuring `PYTHONPATH` as well:

	export PYTHONPATH=~/path/to/scanl_tagger

Finally, you need to install Spiral, which we use for identifier splitting. The current version of Spiral on the official repo has a [problem](https://github.com/casics/spiral/issues/4), so consider installing the one from the link below:

    sudo pip3 install git+https://github.com/cnewman/spiral.git

Finally, we require the `token` and `target` vectors from [code2vec](https://github.com/tech-srl/code2vec). The tagger will attempt to automatically download them if it doesn't find them, but you could download them yourself if you like. It will place them in your local directory under `./code2vec/*`

## Usage

```bash
./main -h                     # Display command options.
./main -v                     # Display the application version.
./main -r                     # Start the server for tagging requests.
./main -t                     # Run the training set to retrain the model.
./main -a [address]           # configure the server address.
./main --port [port]          # configure the server port.
./main --protocol [protocol]  # configure use of http or https
```

`./main -r` will start the server, which will listen for identifier names sent via HTTP over the route:

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
You can train this tagger using the `-t` option (which will re-run the training routine). For the moment, most of this is hard-coded in, so if you want to use a different data set/different seeds, you'll need to modify the code. This will potentially change in the future.

## Errors?
Please make an issue if you run into errors

# Please Cite the Paper!

No paper for now however the current tagger is based on our previous, so you could cite the previous one for now: 

Christian  D.  Newman,  Michael  J.  Decker,  Reem  S.  AlSuhaibani,  Anthony  Peruma,  Satyajit  Mohapatra,  Tejal  Vishnoi, Marcos Zampieri, Mohamed W. Mkaouer, Timothy J. Sheldon, and Emily Hill, "An Ensemble Approach for Annotating Source Code Identifiers with Part-of-speech Tags," in IEEE Transactions on Software Engineering, doi: 10.1109/TSE.2021.3098242.

# Training set
Most of the data used to train this tagger can be found here: https://github.com/SCANL/datasets/tree/master/ensemble_tagger_training_data -- some of it is not there yet.

# Interested in our other work?
Find our other research [at our webpage](https://www.scanl.org/) and check out the [Identifier Name Structure Catalogue](https://github.com/SCANL/identifier_name_structure_catalogue)
