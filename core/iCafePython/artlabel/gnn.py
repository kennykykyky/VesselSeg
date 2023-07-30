from iCafePython.artlabel.artlabel import ArtLabel

def getArtLabel():
    art_label_predictor = ArtLabel()
    return art_label_predictor

# label arteries using GNN model
def GNNArtLabel(art_label_predictor, pickle_graph_name, HR=True):
	art_label_predictor.loadGraphName(pickle_graph_name)
	pred_landmark = art_label_predictor.pred(HR)
	return pred_landmark