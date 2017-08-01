# TODO: imports etc.


def get_tsne_embedding(model_w2v):
    w2v_vectors = model_w2v[model_w2v.wv.vocab]
    tsne = sklearn.manifold.TSNE(n_components=2)
    tsne_vectors = tsne.fit_transform(w2v_vectors)
    return tsne_vectors

def plot_embedding(model_w2v):
    indexed_vocab = {v.index: k for k, v in model_w2v.wv.vocab.items()}
    indexed_vocab_new = sorted([k for k in indexed_vocab.items()], key = lambda x:x[0])
    indexed_vocab_new = [x[1] for x in indexed_vocab_new]

    def plot_embedding_plt(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure(figsize=(50, 50), dpi=300)
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(indexed_vocab[i]),
                     fontdict={'size': 20})

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    def plot_embedding_bokeh(X, width = 1000, height = 1000):
        from bokeh.models import HoverTool
        hover = HoverTool(tooltips = [
                ('Word', '@target'),
                ('index', '$index')
        ])
        
        p = bokeh.plotting.figure(plot_width=width, plot_height=height, tools = [hover, bokeh.models.WheelZoomTool(), bokeh.models.PanTool()])
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['target'] = pd.Series(indexed_vocab_new, index=df.index)

        source = bokeh.plotting.ColumnDataSource(df)
        p.circle(x = df.x, y = df.y, source = source, size=20, color="navy", alpha=0.5)
        return p
    show(plot_embedding_bokeh(tsne_vectors, width = 900, height = 700))
    plot_embedding_plt(tsne_vectors)
    plt.savefig('yes.png')
