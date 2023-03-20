from flask import Flask, request, render_template, jsonify
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save('/home/xuguang/scEMD/src/upload_file/' + file.filename)
    adata = sc.read('/home/xuguang/scEMD/src/upload_file/' + file.filename)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)
    # fig1: expression_distribution_of_genes
    fig1_dir = '/home/xuguang/scEMD/src/static/images/expression_distribution_of_genes.png'
    data = adata.X.A.flatten()
    x_1 = data[(data > 0) & (data <=100)]
    plt.figure(figsize = (10,6),dpi=200)
    plt.title("expression distribution of genes", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('genes', fontdict={'family' : 'Times New Roman', 'size'   : 20})
    plt.xlabel('exprs', fontdict={'family' : 'Times New Roman', 'size'   : 20})
    plt.yticks(fontproperties = 'Times New Roman', size = 18)
    plt.xticks(fontproperties = 'Times New Roman', size = 18)
    plt.hist(x_1, bins=100,  alpha=0.5)
    plt.savefig(fig1_dir)

    # fig2: nFeature_RNA
    fig2_dir = '/home/xuguang/scEMD/src/static/images/nFeature_RNA.png'
    plt.figure(figsize = (10,6),dpi=200)
    plt.title("nFeature_RNA", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('cells', fontdict={'family' : 'Times New Roman', 'size'   : 20})
    plt.xlabel('genes', fontdict={'family' : 'Times New Roman', 'size'   : 20})
    plt.yticks(fontproperties = 'Times New Roman', size = 18)
    plt.xticks(fontproperties = 'Times New Roman', size = 18)
    # plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
    x_1 = adata.obs["nFeature_RNA"]
    n1, bins1, patches1 = plt.hist(x_1, bins=100,  alpha=0.5)
    plt.plot(bins1[:-1],n1,'--')
    plt.savefig(fig2_dir)

    # fig3: UMAP
    fig3_dir = '/home/xuguang/scEMD/src/static/images/umap_leiden.png'
    fig, ax = plt.subplots(figsize = (10,10),dpi=200)
    sc.pl.umap(adata, color=['leiden'], show=False, ax=ax)
    ax.set_title('UMAP with Leiden clustering')
    fig.savefig(fig3_dir)

    return jsonify({
                    "status": "success",
                    "image_url1": fig1_dir[23:],
                    "image_url2": fig2_dir[23:],
                    "image_url3": fig3_dir[23:]
                    })

app.run(debug=True)