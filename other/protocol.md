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
        - 1.1 Applications -> Motivation
        - 2.1 Concepts delete
        - 2.2 Definitions and Notations delete
            - Subsection one level higher
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