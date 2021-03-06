## 11.01.2018
- Thesis
    - Learning to learn
        - In the epigraph to Educational Psycholo.8.,A'Y:Cognitive View, David Ausubel says, "If I had to reduce all of educational psychology to just one principle, I would say this: The most important single factor influencing learning is what the learner already knows. Ascertain this and teach him accordingly" (Ausubel 1968[2nd ed. 1978]).
        - Page 54: "Written or spoken messages are necessarily linear sequences of concepts and propositions.""


## 10.01.2018
- Conversation with Tobias
    - Presentation
        - Conclusion
            - mention other results, summarize them into coherent
    - TuBama and two versions of BA
    - Provide CD/DVD with code and results for UKP
    - Thesis
        - Citation style should be less irritant
            - First author name complete
            - https://de.sharelatex.com/learn/biblatex_bibliography_styles
            - Nikolentzos2017a is wrong
        - Implementation subsection
            - Add grouping step
                - Paper: 3.2
        - When significant, report p-value!
        - Structure of Questions
            - Order of questions? More like a story
            - Question 11: combined -> Question 1
                - Reference other approaches
        - Question 8: Size
            - Pearson correlation
                - Add to table, add assumptions of Pearson r
            - Spearman is appropriate?
            - Use binning, f1 macro
                - Table
                - Show bin interval
                - Show bin size (= number of documents)
                - For coo, cmap and text
                - QCut, remove 99th percentile?
                    - No, keep them
        - Question 11: combined
            - show non-combined scores


## 21.12.2017
- Conversation with Tobias
    - Thesis feedback
        - \phi Visualization
        - ToC
            2.4 - Graph kernel examples not as Section
        - \newpage at new chapter Chapter
        - Table Metrics
            - Padding
        - "d_{istance}"
        - Subgraph
            - E anstatt V
        - Add references
        - Macro Metrics
            - Explain that the macro is over classes
        - Co-occurrence
            - More edges also when word occurs often
            - Example with multiple word occurrence
            - Write to text
        - Graph isomorphism
            - Add citation
        - Hypothesis
            - "Mining" -> leveraging
            - "classification score" -> better word?
            - Define performance in Background
        - BoW
            - Word-Tfidf und Word-Frequency, both are BoW
        - Graph based representations
            - Add "Section"
            - \nameref{}
        - Perceptron
            - Explanation
            - Difference to one-layer net?
            - Perceptron -> One-layer Net
        - Conclusions
            - Limitations
                - What are disclaimers for the work?
                - What did we not analyze despite it being important?


## 18.12.2017
- Thesis
    - Resolved citing issue


## 17.12.2017
- Code
    - Plot for phi distributions improved
    - Added new experiments
        - Relabeling of infrequent labels with combined
- Thesis
    - Wrote text and added plots for phi distribution
    - Added text for infrequent labels experiment
    - Minor grammar check


## 16.12.2017
- Code
    - Added new experiments
        - Relabeling of infrequent labels
    - Plots for node degree distribution
    - Plots for \phi distribution
- Thesis
    - WL example plot improved
    - Related papers: added explanation for _Text classification using Semantic Information and Graph Kernels_


## 15.12.2017
- Break

## 14.12.2017
- Code
    - Cleanup
- Thesis
    - Added text for text-section
    - Added statistics about concept frequency
    - Added example of one-layer neural net
    - Added graph for combined feature SVM analysis


## 13.12.2017
- Thesis
    - Cleanup
    - Added better version of WL examples
    - Added statistics about node degree distribution


## 12.12.2017
- Code
    - Cleanup of embeddings creation
- Thesis
    - Added several examples
        - "Linearization" of graphs
        - Pre-processing
    - Added text to several sections


## 11.12.2017
- Code
    - Added distinction between directed/undirected to concept maps
    - Re-generated w2v embeddings and coref lookup
- Thesis
    - Added example of pre-processing
    - Statistics about edge labels
    - Cleanup

## 10.12.2017
- Code
    - Removed multi-class articles from r8/reuters-21578 dataset
    - Added distinction between one-tail and two-tail permutation test
    - Statistics about correlation of graph/text size and classification performance
    - Statistics about occurrences of edge labels
- Thesis
    - Wrote paragraph about reuters-21578 dataset
    - 


## 09.12.2017
- Code
    - Gathered more graph statistics
    - Cleanup
- Thesis
    - Added paragraph for nyt dataset


## 08.12.2017
- Code
    - NYT dataset
        - Initial exploration
        - Extracted meta-data, body, ...
        - Category and word count filter
    - Cleanup


## 06.12.2017
- Skype call with Tobias
    - Merge labels of un-frequent concepts
    - New York Times corpus to concept maps, then classification


## 03.12.2017
- Thesis
    - Cleanup
    - Idea/ToDo
    	- Add score per graph size analysis we did before
	    - **IMPORTANT**: Do not forget to add standard deviations to result tables!
- Code
    - Improved README and bootstrap scripts (to install dependencies, download datasets, ...)
    - Added statistics about WL similarities per iteration
- Idea/ToDo
    - Problem: some concepts only occur once per dataset, no chance of match
    - But...
    - labels which only occur once in train set, but also in test set, would then not be found
    - Possible solutions
        - remove these concepts
        - relabel with similar label
        - we have tried this before but for _all_ labels, not only for the less frequent labels
 

## 02.12.2017
- Thesis
    - Wrote section for related paper _"Text classification using Semantic Information and Graph Kernels"_
        - DRS graphs
        - custom node equality function for random-walk graph kernel


## 01.12.2017
- Code
	- Fixed LaTex code
- Thesis
	- Wrote section for "Deep Graph Kernel" paper (related work)
		- Find similarities in substructures
		- ... create similarity matrix and multiply phi feature map when calculating the kernel
		- k(G, G') = phi(G)^T * M * phi(G')
			- M = similarity matrix of substructures counted in phi feature map
	- Read "Text classification as a graph classification" paper, again
		- Frequent subgraph mining
		- Co-occurrence graphs
		- "Long-distance n-grams"
		- Support value
			- number of times a subgraph appears in dataset
- Ideas
	- Concept map multi-label lemmatization
		- Lemmatize the single labels after splitting to increase match probability


## 30.11.2017
- Laptop charger died
	- Had to setup new laptop
- Started concept map extraction for more datasets
- Code cleanup


## 29.11.2017
- Code
    - ted_talks dataset
        - Created co-occurrence graphs
        - Tried converting multi-labels to single-labels
    - created w2v embeddings for relabeling
- Ideas
    - Full-text datasets
        - PubMed
            - Only citations
            - http://compbio.ucdenver.edu/ccp/corpora/obtaining.shtml
        - https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-207
        - http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0039464

## 28.11.2017
- Code
    - Splitting multi-word labels into separate nodes
        - Questions
            - Add self links?
            - Remove stopwords?
            - Keep directionality?
            - Keep edge labels?
        - Tests
            - Split labels vs. non-split
                - Far better performance with splitted labels!
            - Combined graph+text features with splitted vs non-splitted
                - No performance improvement
    - Added same_label parameter to fast_wl
- Ideas
    - Quantify probability of matches per WL iteration
        - Steps
            - Look at feature maps of WL iterations
            - ... create dot products
            - ... compare the results for each WL iterations (sum)
        - The sums of dot products of feature maps corresponds to the number of matches


## 27.11.2017
- Code
    - Started concept map creation for datasets with new preprocessing
    - Created ted_talks dataset
        - Clustering of tags
- Thesis
    - Wrote paragraph about Tobias' code for concept map extraction


## 26.11.2017
- Thesis
    - Added paragraph about WL extension
- Code
    - Cleanup
    - Added more experiments
    - Save best classifier away


## 25.11.2017
- Added experiments pipeline
    - Experiments can be defined with yaml files
        - (nearly) all parameters can be defined
- Observations/Ideas
    - Gram matrix reduces information of each object to N numbers where N is the number of objects in the dataset
        - ... but size increases approximately by factor 3
- How to use the multi-word labels of concept maps?
    - Approaches
        - Split multi-word labels, create nodes and add original edges to new word nodes
            - This will increase the graph size
            - ... but also make exact matches far more difficult in later iterations
- Look at sparsity of text- vs. graph-features
    - Hypothesis
        - WL graph features are more sparse than text features
    - Experiment
        - ng20, concept maps WL feature sparsity vs. text feature sparsity
            - sum(all_graph_features) / sum(all_text_features) = 0.57
            - ie. graph features are more sparse than text features
            - that must be taken into account when looking at the contribution of graph features when combining text- and graph features
- Datasets with long texts
    - Gutenberg dataset
        - https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html
        - Problem
            - Mostly belletristic texts?
            - so, no technical texts, most likely no recurring and mineable concepts/relations
    - Textbooks?
        - http://bookboon.com/en
        - http://www.openculture.com/free_textbooks
        - http://textbookrevolution.org/index.php/Main_Page
        - http://www.freebookspot.es/
        - http://manybooks.net/
        - https://www.getfreeebooks.com/
        - http://freecomputerbooks.com/
        - http://www.freetechbooks.com/
        - http://www.onlineprogrammingbooks.com/
        - https://www.bookyards.com/en/categories
        - http://onlinebooks.library.upenn.edu/subjects.html
            - Problem: difficult to crawl since different formats
        - http://www.baen.com/catalog/category/view/s/free-library/id/2012
        - http://www.ebooklobby.com/
    - Wikipedia
        - http://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/
            - Very big (english 4,25GB)
    - Scientific papers


## 24.11.2017
- Worked at imsdb dataset


## 23.11.2017
- Feedback von Karsten
    - Mehr konzeptuelle Fragen
        - zB. was ist mit Multilabels
- Reworked pre-processing
- Tested
    - WL with ignore_label_position=true
        - ignore_label_position seems to be slightly better (1-2%)
    - Tried to further increase combined performance
        - Test more C parameters
        - Test l1 regularization
            - Need to optimize primal not dual problem
                - This is a constraint for l1 regularization with sci-kit learn
            - Performance is decreasing dramatically (~20%)
    - Looked at coefs from classifier at combined features case
        - Problem: with SVM regularization
            - l1 with dual=true is not possible
            - l2 with dual=False is possible
                - ... but performance is far worse than with dual=True
            - Solution:
                - Use other classifier like perceptron, SGD, ...
                - ... that also uses coefficients that are a indicator for the importance of a feature
        - Different regularizations
            - l1
            - l2
        - Sum of text-/graph-features 50/50
            - But: graph features are most likely more sparse
        - Graph feature coefs are distributed nearly uniformly across iterations
- Extensions to WL?
    - Node weights as factors to WL
        - PageRank or degrees as factors to multiply the fields in WL with
        - Intuition
            - With greater iterations, exact matches of large neighbourhoods are difficult, so...
            - ... prioritize such matches by mulitiplicating with the degree or importance under PageRank
    - Ignore label position
        - Normal WL: given some node, create new label with (label(current_node) + sorted([label(node) for node in neighbours(current_node)]))
        - Without label position: sorted([label(node) for node in neighbours(current_node) + label(current_node)]
        - How similar is this to orderless n-grams?
            - ie. counting sets of labels per node
    - "Residual WL"
        - Not thought through
        - Remove labels from last iteration
            - the features of iteration I only take nodes that are I hops away into account
- Idea
    - To show correlation between classes eg. in ng20
        - ... vectorize text, eg. BoW
        - ... create embeddings with tSNE
        - ... plot
- Searched new, long datasets
    - imsdb
        - Create single labels from multi label by clustering


## 22.11.2017
- Added other fast_wl implementation to cross-test results
- Search for larger dataset
    - Constraints
        - High quality content (= most likely no user generated comments)
            - Technical (?) with recurring concepts
        - Long texts per doc
            - over 1000 words
        - Concatenate text with each other, generate and look at resulting CMs 
    - Candidates
        - TED talk transcripts
            - Problem: no labels, but tags
                - Tags could be merged by co-occurrence
                - Non-frequent tags are removed
        - Movie transcripts or subtitles
            - Example
                - http://www.imsdb.com/all%20scripts/
                - http://www.simplyscripts.com/
                - https://sfy.ru/
                - https://github.com/agonzalezro/python-opensubtitles
            - Genre as labels
            - Problem: Parsing
            - Problem: most likely no coherent
        - The New York Times Annotated Corpus
            - https://catalog.ldc.upenn.edu/ldc2008t19
            - https://catalog.ldc.upenn.edu/docs/LDC2008T19/new_york_times_annotated_corpus.pdf
            - Stats
                - 1,855,658 documents
                - January 01, 1987 and June 19, 2007
            - No explicit word count per document given
                - ... but most likely is very high since they are articles
            - Problem: costs money ($150-300)
                - ... but seems to be a good dataset
        - Papers


## 21.11.2017
- Looked at recurring concepts in concept maps per dataset
    - How many concepts appear only once in whole dataset?
    - Compare to co-occurrence
- Other ToDos
    - Remove concepts only occurring once
    - Find corpus with long texts
        - TED talk transcripts?
            - How to obtain labels? 
                - From tags?
    - Gather results for binarized CountVectorizer vs un-binarized (= frequencies)
    - Gather more results on node weighting with WL
    - Find explanation for lower performance when combining features
        - Regularization with l1
            - Quantify difference
    - Create significance tests for models?
        - Which ones?
    - Precisely define experiments for each subquestion
    - Test more normalization on graph features
    - Find differences in datasets
        - How many labels are unused in test phase
        - Compression factor of highest-phi-index/#vertices
        - When does WL converge for co-occurrence and concept maps


## 20.11.2017
- Created more statistics about datasets
- Visualized SVM coefficients of combined features
- Looked for more datasets with long texts


# 19.11.2017
- Searched for longer datasets


# 18.11.2017
- Co-Occurrence fix
- Searched for longer datasets
- Cleanup


# 17.11.2017
- Tested more C combinations for SVM
- Added early stopping to fast_wl
- Rewriting code to use nested cross-validation more easily
- Cleanup


# 16.11.2017
- Added node weights to WL
    - pagerank
    - degrees
- Started rewriting pipeline code

## 26.10.2017
- Slides about goals/results/observations
    - max 5 slides
    - Also progress on non-finished goals/questions
- Significance test
    - Exact/Randomization test Implementation
        - For confidence: only count diffs that are greater than global diff?
    - Reference implementation?
        - Test against student-t test, should be consistent
    - Confusion Matrix of gram matrix similarities between classes
        - mean/std and student-t test
- Thesis
    - ToC
        - 1
            - 1.1 Applications -> Motivation
            - 2.1 Concepts delete
        - 2
            - 2.2 Definitions and Notations delete
                - Subsections one level higher
                - Concept Map and Co-Occurrence Graph as subsections to "Graphs"
        - Text classification
        - Related Work -> "Graph kernel based text classification"
        - Split "Evaluation" into
            - Experiments and Explanations of Approach ("Experimental Setup")
            - Results and Discussion (Observations)
        - Implementation chapter as child of "Experiments"
        - Enumerate questions in "Experimental Setup" for later reference
    - Content
        - Hypothesis and Goals: remove subgoals


## 28.08.2017
- Problem: SVMs learn better with zero mean and unit variance data
    - Sparse feature vectors (like phi or tfidf vectors) will become dense after subtracting mean (= zero mean)
    - Solution: use SVM kernel without the need of input centering
        - linear, poly, ...


## 23.08.2017
- Co-reference resolution
    - Problem: Composite Labels (= Labels mit mehr als einem Wort)
    - Problem: Nach Co-Reference-Resolution kann es sein, dass Graph zweimal das gleiche Label hat
        - Lösung: Adjazenzmatrix muss angepasst werden
    - Problem: Word2Vec Embeddings nur auf Train-Set trainieren. Rechenzeit erhöht sich


## 22.08.2017
- Resultate hinzugefügt
    - https://github.com/davidgengenbach/bachelor-thesis/blob/master/code/tmp/results/README.md


## 21.08.2017
- Problem: Wann sind zwei Labels gleich
    - Lösung implementiert
        - Tresholding
            - Es gibt zwei Embeddings
                - ein pre-trained (GloVe, Google News)
                - ein selbst auf dem Datenset trainiertes Word2Vec Embedding
            - 1) Labels werden aufgelöst, die im pre-trained Embedding fehlen
                - Für im pre-trained Embedding fehlende Labels wird im selbst-trainierten Embedding nach ähnlichen Wörtern gesucht...
                - ... dann wird das Embedding für das ähnliche Label für das fehlende Label benutzt
            - 2) Label-Cliquen werden gebildet
                - Für jedes Label wird das ähnlichste Label gesucht
                - Wenn die (cosine) Ähnlichkeit über einem Treshold ist, kommen die beiden Labels (ursprünglich + ähnlich) in die selbe Clique
                - siehe https://github.com/davidgengenbach/bachelor-thesis/blob/master/code/coreference.py#L5


## 20.08.2017
- Problem: Wann sind zwei Labels gleich?
    - Wie groß darf der Cosine Abstand sein, damit zwei Labels gleich sind? 
    - Mögliche Lösung
        - Clustering
            - zB. K-Means auf die Word2Vec Embeddings der Labels mit K Cluster (= Anzahl Bins). Alle Labels in einem Cluster bekommen dasselbe (neue) Label
            - Contra
                - hoher Rechenaufwand
                - Wie Anzahl der gewünschten Cluster bestimmen?
        - Treshold
            - Alle Labels miteinander vergleichen. Wenn (cosine) Abstand unter Treshold, werden zwei oder mehr Labels zusammengefasst
            - Wie Treshold bestimmen?
                - manuell angeben
                - pro Datenset berechnen
                    - zB. Median-Abstand von Embeddings durch festgelegte Zahl?
            - Implementierung
            - Contra
                - da transitiv, gibt es wahrscheinlich große Cluster
        - Most similar
            - "Natürliches Clustering"
            - Contra
                - da transitiv, gibt es wahrscheinlich große Cluster


## 19.08.2017
- Problem: fehlende W2V Embeddings für Graph-Labels bei *Concept Maps*
    - (Mögliche) Lösungen
        - Ignorieren
        - Labels aufsplitten (manche Labels bestehen aus 2 oder mehr Wörtern), dann jeweils die Embeddings davon aufaddieren (und average)
        - Andere Co-reference Auflösung
        - selbst trainierte W2V Embeddings vom Ausgangsdatenset verwenden, um ähnliche Wörter zu den Labels zu finden, die dann in pre-trained (GloVe, Google) W2V suchen


## 18.08.2017
- Phi Cache für Datensets eingebaut, um Klassifizierung zu beschleunigen
- Klassifizierungsergebnisse für Datensets zusammengesammelt
    - müssen noch harmonisiert werden
        - (anscheinend) gibt sklearn die Params für GridSearchCV unterschiedlich aus
- Fehlende W2V Embeddings
    - W2V Embeddings für alle Datensets generiert
    - "trained" Embeddings: auf den einzelnen Datensets selbst trainierte Word2Vec Embeddings
    - Recht viele fehlende Embeddings, die aufgelöst werden müssen
    - bei Concept Maps
        - Fehlende Labels (Mean über alle Datensets)
            - GoogleNews-vectors-negative300    93%
            - glove.42B.300d                    91%
            - glove.6B.100d                     92%
            - glove.6B.200d                     92%
            - glove.6B.300d                     92%
            - glove.6B.50d                      92%
            - glove.840B.300d                   92%
            - glove.twitter.27B.100d            92%
            - glove.twitter.27B.200d            92%
            - glove.twitter.27B.25d             92%
            - glove.twitter.27B.50d             92%
            - trained                           91%
            - (total)                           92%
        - ... liegt wohl daran, dass die Label in den Concept Graphs nicht nur ein Wort beinhalten
            - (mögliche) Lösung
                - Wörter splitten ...
                - ... dann Word2Vec für Einzelwörter
                - ... dann Mean von den Vektoren der Einzelwörter
    - bei Co-occurence graphs
        - Fehlende Labels (Mean über alle Datensets)
            - GoogleNews-vectors-negative300    47%
            - glove.42B.300d                    27%
            - glove.6B.100d                     40%
            - glove.6B.200d                     40%
            - glove.6B.300d                     40%
            - glove.6B.50d                      40%
            - glove.840B.300d                   27%
            - glove.twitter.27B.100d            46%
            - glove.twitter.27B.200d            46%
            - glove.twitter.27B.25d             46%
            - glove.twitter.27B.50d             46%
            - trained                           29%
            - (total)                           39%
        - mögliche Gründe für fehlende Labels
            - bei den pre-trained Embeddings
                - Rechtschreibfehler: ein Teil der Datensätze kommt direkt aus Foren/Emails
                - Zu spezielle Wörter
            - bei den selbst trainierten Embeddings
                - ???
                - Hier dürfte es keine fehlenden Labels geben!


## 17.08.2017
- Input von Tobias
    - Vielleicht pre-trained GloVe Embeddings anstatt pre-trained Word2Vec
        - Anzahl der fehlenden Labels?
- Word2Vec Ansatz für Label-Binning
    - Problem: Beim pre-trained GoogleNews Word2Vec fehlen viele Labels vom Datensatz
        - Mögliche Lösung
            - eigene Embeddings über Word2Vec lernen auf Datensatz (mit allen Labels) ...
            - ... dann fehlende Labels durch ähnliche Labels ersetzen, die im pre-trained GoogleNews Word2Vec sind
            - ... dann normal weiter machen
    - Problem: Lemmatized Labels sind nicht in Embeddings bei den Concept-Maps
        - Die Labels in den Concept Maps sind lemmatized, und sind wahrscheinlich nicht in pre-trained Embeddings von Google w2v
        - (Doch kein Problem, da die Labels _nicht_ lemmatized sind)


## 15.08.2017
- Ansatz: Word2Vec der Graph Labels
    - Problem: für viele Labels gibt es kein Embedding im Google News Datensatz 
    - (mögliche) Lösung
        - Word2Vec auf ganzen Datenset selber trainieren


## 14.08.2017
- Ergebnisse (vorerst!)
    - Höhere Iterationen von WL -> niedrigerer F1 score
        - Mögliche Erklärung
            - Damit zwei Graphen ein nicht-null Dot-Produkt haben, muss es in beiden auch Nachbarschaften geben, die exakt (!) die gleichen Knoten haben
            - Bei der nullten Iteration von WL werden nur die Labels gezählt, also sind sich zwei Graphen schon ähnlich, wenn sie gleiche Labels (= Wörter) im Text haben
- Neuer (?) Ansatz
    - Word2Vec lernen auf allen Dokumenten des Datensets...
        - ... dann Ähnlichkeit der Labels über die gelernten Embeddings
        - ... dann Binning der Labels -> zwei oder mehr ähnliche Labels werden zu einem Label zusammengefasst
        - Vorteil: weniger Rechenzeit
            - da die Embeddings wahrscheinlich kleiner sein können
            - und nur relevante Labels gelernt werden
            - (Nicht relevant, da es O(n^2) Vergleiche gibt - das dominiert gegenüber dem linearen Speedup)
    - Bestehende Word2Vec Embedder benutzen (Google)...
        - (weiteres dann gleich wie oben drüber)
        - Nachteil: höhere Rechenzeit
            - Bei großen Datensets gibt es bei N unique Labels dann N(N-1) Kombinationen, die getestet werden müssen


## 13.08.2017
- fast_wl
    - implemented fast hashing-based WL
    - (Wirklich schneller, numpy optimizations/shortcuts selbst hinzugefügt)
    - mit alter Implementierung vergleichen!
        - F1 und Performanz
        - Korrektheit
- Statistiken
    - Dateigrößen
        - Zusammenhang Window Size <-> Dateigröße
- Beobachtungen
    - Vorerst!
        - mehr Iterationen von WL -> niedriger F1
        - Vergleich Co-occurence vs Concept Maps
            - Co-occurence grundsätzlich höherer F1
            - Graphen größer bei Co-occurence, Beispiel:
                - dataset_graph_cooccurrence_1_all_ng20.npy
                    - Window Size: 1
                    - Alle Wörter
                    - 126MB
                - dataset_graph_cooccurrence_1_only-nouns_ng20.npy
                    - Window Size: 1
                    - Nur Nomen
                    - 38MB
                - dataset_graph_gml_ng20-single.npy
                    - Concept-Map
                    - 31MB
                - dataset_ng20.npy
                    - Als Text
                    - Alle Wörter
                    - 22MB


## 10.08.2017 (Treffen)
- Graphen
    - "no-nouns" -> "only-nouns"
- Results
    - alle Parameter mit Ergebnissen in Tabelle
- Co-reference
    - GloVe Embeddings


## 06.08.2017
- Unterhaltung Prof. Kersting per Skype
    - Cross-Validation benutzen
    - Hyperparameter Tuning durch GridSearch, ...
    - Gewichtung der Edges durch Word2Vec
        - Bei DeepWalk?
        - Aufaddieren der Word2Vec Vektoren von DeepWalk Embedding als Repräsentation von Dokument
    - Fast_WL benutzen?
        - https://github.com/rmgarnett/fast_wl
- Cross-Validation und Hyperparameter Tuning
    - Hyperparameter Tuning eher stochastisch anstatt Grid?
        - http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
    - Neuen Transformer für verschiedene Classifier
    - Preprocessing Transformer für sklearn schreiben
- WL Kernel Transformer
    - Batching?
- Fast WL
    - Funktion um neues Label/Color für einen Node zu finden ist:
        - l_new(x) = l_old(x) + sum_{y element neighbours(x)}:(log(prime_number_with_index(label_old(y))))
    - Frage ist, wo verschnellerung ist, wenn sowieso keine String-Vergleiche gemacht werden, sondern ein Label Lookup über die IDs der Labels gemacht wird
- Graph Convolutional Networks
    - https://tkipf.github.io/graph-convolutional-networks/
    - http://www.inference.vc/how-powerful-are-graph-convolutions-review-of-kipf-welling-2016-2/
    - Paper: "Semi-supervised Classification With Graph Convolutional Networks"
        - https://arxiv.org/pdf/1609.02907.pdf
- word2vec gradient through document?


## 05.08.2017
- DeepWalk
    - per Graph
    - generiert Random-Walks auf dem Graph
    - trainiert Word2Vec mit den Random Walks, um Vektor-Repräsentationen für Nodes zu finden
    - Wie DeepWalk für Text-Klassifikation verwenden?
- Code in sklearn Pipeline umschreiben (Klassen, die 'transform' methode haben)?


## 04.08.2017
- Datensets
    - Graph dataset retrieval vereinfacht (wird gecached)
    - Vielleicht alle Zwischenstufen speichern? (pre-processing, co-occurence graph, ...)
        - Derzeit werden nur Endprodukte gespeichert (Co-occurence graph)
    - Graphen
        - Mehr Statistiken sammeln
            - Most-frequent node label
            - Most-frequent edge label
            - Sparsity/Density
            - Avg labels per dataset
            - Reduction to initial dataset
- Kernel
    - WL
        - Phi Erweiterung: Tests schreiben
        - Weighting of phi by norm?
- Classifier
    - Class weighting
    - Normalize Phi!
    - Combine classical and graph vector
- Other graph embedding/... approaches
    - Deep Walk
        - by Perozzi
        - Paper: ""
            - http://perozzi.net/publications/14_kdd_deepwalk.pdf
    - LINE
        - by Tang
        - Paper: "LINE: Large-scale Information Network Embedding"
            - http://dl.acm.org/citation.cfm?id=2736277.2741093
    - GraRep
        - by Cao
        - "... a novel model for learning vertex representations of weighted graphs."
        - Paper: "GraRep: Learning Graph Representations with Global Structural Information"
            - http://dl.acm.org/citation.cfm?id=2806512
        - Repo: https://github.com/ShelsonCao/GraRep
    - GGSNN
        - by Li et al
        - Paper: "Gated Graph Sequence Neural Networks"
            - https://arxiv.org/pdf/1511.05493.pdf
    - CENE
        - by Sun et al
        - Paper: "A General Framework for Content-enhanced Network Representation Learning"s
            - https://arxiv.org/pdf/1610.02906.pdf
    - Community Preserving Network Embedding
        - by Wang et al
        - Paper: "Community Preserving Network Embedding"
    - Fast Network Embedding Enhancement via High Order Proximity Approximation
        - Paper
            - http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2017_fastnrl.pdf
    - https://github.com/thunlp/NRLPapers


## 03.08.2017
- Kernel
    - Most frequent Subgraphs
        - Only for recurring subgraphs - not interesting for co-occurence or concept maps
- Co-occurence
- Pre-processing


## 03.08.2017 (Treffen)
- Kernel
    - Graph product implementieren
    - Vielleicht most-frequent subgraph mining (K-core etc.)
- Co-occurence
    - In einheitliches Format übertragen
    - Un-directed
    - How to prune graph?
        - Nur Node hinzufügen, wenn höchstens x mal im Text
        - Part-of-speech, dann nur Nomen benutzen (spacy)
        - So stark verkleinern wie möglich, dann Ergebnisse vergleichen
            - Parameter (Window Size, POS tagging)
- Pre-processing
    - "Substitute TAB, NEWLINE and RETURN characters by SPACE."
- Datasets
    - r-21578 zu r-8 (direkt mit Graphen)
- Graph statistics
    - Histogramm #nodes, Achse?


## 02.08.2017
- Zeiten
    - Co-occurence
        - ng20
            - retrieve dataset and pre-processing
                - 20s
            - generate co-occurence graphs
                - 1min 30s
            - convert to networkx graph + save to gml
                - 1min 40s
- Coding
    - Co-occurence graphs
        - Directed or un-directed?
        - Graphs als npy speichern, um GML parsen zu vermeiden
        - How to prune graph?
    - Pre-processing
        - lowercase
        - remove "more than one" linebreaks
        - (optional) remove stopwords
        - (optional) remove interpuncation
    - Graph statistics


## 30.07.2017
- Drei Paper von Tobias
    - Text classification using Semantic Information and Graph Kernels
        - http://www.di.uevora.pt/~pq/papers/epia2011a.pdf
        - Datasets: Reuters
    - Concept Graph Preserving Semantic Relationship for Biomedical Text Categorization
        - http://www.researchpublications.org/IJCSA/NCRMC-14/08.pdf
        - Datasets: Mediline
    - Text Categorization as a Graph Classification Problem
        - http://www.aclweb.org/anthology/P15-1164
        - Datasets:
            - WebKB: 4 most frequent categories among labeled webpages from various CS departments – split into 2,803 for training and 1,396 for test (Cardoso-Cachopo, 2007, p. 39–41).
                - Ana Cardoso-Cachopo. 2007. Improving Methods for Single-label Text Categorization. Ph.D. thesis, Instituto Superior Técnico, Universidade de Lisboa, Lisbon, Portugal.
                - http://web.ist.utl.pt/~acardoso/docs/2007-phd-thesis.pdf
            - R8: 8 most frequent categories of Reuters- 21578, a set of labeled news articles from the 1987 Reuters newswire – split into 5,485 for training and 2,189 for test (Debole and Sebastiani, 2005).
                - Franca Debole and Fabrizio Sebastiani. 2005. An Analysis of the Relative Hardness of Reuters-21578 Subsets: Research Articles. Journal of the American Society for Information Science and Technology, 56(6):584–596.
                - http://onlinelibrary.wiley.com/doi/10.1002/asi.20147/full
            - LingSpam: 2,893 emails classified as spam or legitimate messages – split into 10 sets for 10-fold cross validation (Androutsopoulos et al., 2000).
                - Ion Androutsopoulos, John Koutsias, Konstantinos V. Chandrinos, George Paliouras, and Constantine D. Spyropoulos. 2000. An Evaluation of Naive Bayesian Anti-Spam Filtering. In Proceedings of the Workshop on Machine Learning in the New Information Age, 11th European Conference on Machine Learning, pages 9–17.
                - https://arxiv.org/abs/cs/0006013
            - Amazon: 8,000 product reviews over four different sub-collections (books, DVDs, electronics and kitchen appliances) classified as positive or negative – split into 1,600 for training and 400 for test each (Blitzer et al., 2007).
                - John Blitzer, Mark Dredze, and Fernando Pereira. 2007. Biographies, bollywood, boomboxes and blenders: Domain adaptation for sentiment classi- fication. In Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics, ACL ’07, pages 440–447.
                - http://anthology.aclweb.org/P/P07/P07-1.pdf#page=478
                - https://www.cs.jhu.edu/~mdredze/datasets/sentiment/


## 28.07.2017
- Datasets
    - WebKB
        - Webpages with categories
        - http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/
        - Download
            - http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz
    - Reuters
        - Reuters-21578
            - http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html
            - Download
                - http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz
        - RCV1-v2
            - Sources (= Text) NOT available, only feature vectors
            - http://www.daviddlewis.com/resources/testcollections/rcv1/
            - Download
                - http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a01-list-of-topics/rcv1.topics.txt
                - http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig
                - http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a03-expanded-topics-hierarchy/rcv1.topics.hier.expanded
    - LingSpam
        - http://csmining.org/index.php/ling-spam-datasets.html
        - Download
            - http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public.tar.tar
            - http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public01.tar.tar
            - http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public02.tar.tar
    - Amazon reviews


## 27.07.2017 (Treffen)
- Prediction Error: "Wieso eine Klasse oft mispredicted?"
    - Skewed labels?
    - von Graphen-Größe abhängig?
- Statistiken der Dokument/Graphen-Größe
- Neue Datensets
    - An Tobias schicken (Namen, Link)
- Co-Occurence Graphs auch benutzen
- Bei SGDClassifier Features normalisieren
- Mehrere Classifier bei Graphen
- Alle Dokumenten-Graphen pro Thema in einem Graph?
    - Train-Split an Tobias schicken (gleichviele Dokumente pro Topic)
- Co-Referenz Auflösung bei WL zwischen Node-Labels
    - Word-Embeddings per w2v: Feature Vektoren von Labels (cosine)
- Limitierung: Alle Labels müssen zu Beginn bekannt sein
- Paper von Tobias lesen
- Abschlussarbeit Formular ans Studiensekretariat


## 20.07.2017 (Treffen)
- Abschlussarbeit Formular Studiensekretariat
    - Macht Tobias
- Dokumente an Tobias schicken
    - Tobias berechnet Graphen
- Kernel genauer anschauen
- Warum SVM, wenn Kernel similarity zwischen Graphen ausgibt?
- Ab und zu Dokument mit Ergebnissen/Status (als GitHub)
- Co-Occurence Graph
    - Bestehenden Code mit Window Size?
    - Mit oder ohne Gewichtung?
    - Cut-Off danach?
    - Was ist ein Knoten? Part of Speech, Stopwords rauswerfen
- Bei Vergleich zwischen Baselines immer das Beste
- Vielleicht Visualisierung
    - Antoine Jean-Pierre Tixier
- https://github.com/ipython/talks/blob/master/parallel/text_analysis.py


## 19.07.2017
- How to speed-up gram-matrix calculation?
- k-fold cross-validation with graphs?
- Server!
- Weisfeiler-Lehman
    - Input
        - n Graphs with...
        - adjacency matrices: n x Array(m, m)
        - node_labels: n x Array(SOME_NUMBER_OF_LABELS)
    - Gram Matrix


## 14.07.2017
- PipelineGroupedConceptRecall
    - textPattern/mapName auch anpassen
    - String[] pipeline = { "extraction.PropositionExtractor", "grouping.ConceptGrouperBase", "grouping.ExtractionResultsSerializer", "extraction.AllConceptsMap" };
        - String[] pipeline = { "extraction.PropositionExtractor", "grouping.ConceptGrouperSimLog", "grouping.ExtractionResultsSerializer"};
- Später schauen, ob man die Pipeline erweitert
- Workflow
    - PipelinePreprocessing
    - PipelineOpenIE
    - PipelineGroupedConceptRecall
    - scoring.concepts.features.ExportGraphs


## 13.07.2017 (Treffen)
- Idee
    - Concept Maps als Input für Text-Klassifikation, Vergleich gegen andere Repräsentationen
- Experimente:
    - Hypothese
        - Concept Map Graphen sind bessere Repräsentation für die Klassifizierungsaufgabe
    - Daten
        - Text-Klassifikation (Paare Dokument + Klassenlabel)
            - Start: 20newsgroups
            - Weitere: anschließend entscheiden
    - Methoden
        - Baselines
            - Bag of Words -> n-gram Häufigkeit in Standard-Classifier (Logistic Regression, Bayes, SVM, etc.)
            - Bag of Words, aber word embeddings statt n-grams, dann simples MLP (= Multilayer Perceptron)?
            - Word Co-Occurrence-Graph, dann Graph-Verfahren
        - Aus Dokumenten extrahierte Concept Maps, dann Graph-Verfahren
    - Graph-Methoden (für graph-basierten Input)
        - Verschiedene Graphkernel
        - Graph Deep Learning
    - Implementierung
        - Pointer für existierende Graph-Methoden
            - Notizen von Kristian
        - Extraktion von Concept Maps
            - Von Tobias
- Orga
    - Dokumentation der Ergebnisse in Dropbox o.ä.