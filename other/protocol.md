## 10.08.2017 (Treffen)
- Graphen
    - "no-nouns" -> "only-nouns"
- Results
    - alle Parameter mit Ergebnissen in Tabelle
- Co-reference
    - Glove Embeddings


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